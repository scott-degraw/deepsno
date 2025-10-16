#include <RAT/DS/Entry.hh>
#include <RAT/DS/PMT.hh>
#include <RAT/DU/DSReader.hh>
#include <RAT/DU/Utility.hh>
#include <RAT/PMTSelector.hh>
#include <RAT/PMTSelectorFactory.hh>
#include <RAT/FitterPMT.hh>
#include <RAT/PMTCalib.hh>
#include <cassert>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <system_error>
#include <vector>
#include <array>
#include <limits>
#include <memory>
#include <TEntryList.h>
#include <TFile.h>
#include <TCut.h>
#include <TSystem.h>
#include <TParameter.h>
#include <TVector3.h>
#include <ROOT/RDataFrame.hxx>

std::vector<std::string> ntuple_branches = {
    "runID", "eventID", "nhits", "fitValid", "posx", "posy", "posz", "posz_av", "posr_av", "energy", "time", 
    "nhitsCleaned", "nearAV", "itr", "necknhits", 
};

std::vector<std::string> mc_ntuple_branches = {
    "mcke1",
    "mctime1",
    "mcPosx",
    "mcPosy",
    "mcPosz",
};

constexpr Float_t kFloatNaN = std::numeric_limits<Float_t>::quiet_NaN();
constexpr Double_t kDoubleNaN = std::numeric_limits<Double_t>::quiet_NaN();

struct PmtEvent {
    std::vector<UInt_t> id;
    std::vector<Float_t> hit_time;
    std::vector<Float_t> qhs;
};

struct PmtInfo {
    std::vector<Float_t> position;
};

struct McEvent {
    std::vector<Float_t> times_of_flight;
    Float_t global_trigger_time = kFloatNaN;
};


void ratds_extract(std::string input_fname, 
        std::string ntuple_fname,
        std::string output_fname,
        Float_t min_ht, Float_t max_ht, Float_t min_qhs, Float_t max_qhs, 
        std::string filter = "", bool eca_cal = false) {
    try {
        std::cout << "Extracting data from " << input_fname << " into " << output_fname << "\n";

        if (gSystem->AccessPathName(input_fname.c_str(), kFileExists) != 0) 
            throw std::runtime_error("Input file " + input_fname + " does not exist");
        if (gSystem->AccessPathName(ntuple_fname.c_str(), kFileExists) != 0) 
            throw std::runtime_error("Ntuple file " + ntuple_fname + " does not exist");

        std::cout << "Using ntuple file " << ntuple_fname << "\n";

        TFile ntuple_file(ntuple_fname.c_str(), "READ");
        TTree * ntuple = ntuple_file.Get<TTree>("output");

        // Start the data reading 
        RAT::DU::DSReader dsreader(input_fname);
        dsreader.BeginOfRun();

        auto run_info = dsreader.GetRun();
        bool is_mc = run_info.GetMCFlag();

        RAT::DB *db = RAT::DB::Get();

        TFile output_file(output_fname.c_str(), "RECREATE");
        TTree pmt_info_tree("pmt_info", "Contains PMT information");

        ntuple->SetBranchStatus("*", 0); // Disable all branches
        for (const auto &branch : ntuple_branches) {
            ntuple->SetBranchStatus(branch.c_str(), 1); // Enable only the branches we need
        }
        if (is_mc) {
            for (const auto &branch : mc_ntuple_branches) {
                ntuple->SetBranchStatus(branch.c_str(), 1); 
            }
        }
        TTree * event_tree = ntuple->CloneTree(0);
        event_tree->SetDirectory(&output_file);
        event_tree->SetTitle("Contains event level data");
        event_tree->SetName("event");

        Int_t run_id;
        Int_t event_id;
        ntuple->BuildIndex("runID", "eventID");

        std::vector<Double_t> av_offset_vec = db->GetLink("GEO", "av")->GetDArray("position");
        if (av_offset_vec.size() != 3) {
            throw std::runtime_error("AV offset vector should have 3 elements, but has " + std::to_string(av_offset_vec.size()));
        }
        std::array<Double_t, 3> av_offset;
        std::copy(av_offset_vec.begin(), av_offset_vec.end(), av_offset.begin());

        event_tree->Branch("av_offset", &av_offset);

        RAT::DBLinkPtr native_geo_dims_link = db->GetLink("NATIVE_GEO_DIMENSIONS", "natgeo_dimensions");
        Double_t inner_av_radius = native_geo_dims_link->GetD("inner_av_radius");
        Float_t av_thickness = native_geo_dims_link->GetD("av_thickness");

        TParameter<Float_t> param_inner_av_radius("inner_av_radius", static_cast<Float_t>(inner_av_radius));
        TParameter<Float_t> param_av_thickness("av_thickness", static_cast<Float_t>(av_thickness));
        param_inner_av_radius.Write();
        param_av_thickness.Write();

        PmtEvent pmt_event;
        McEvent mc_event;

        event_tree->Branch("pmt_id", &pmt_event.id);
        event_tree->Branch("pmt_hit_time", &pmt_event.hit_time);
        event_tree->Branch("pmt_qhs", &pmt_event.qhs);
        if (is_mc) {
            event_tree->Branch("mc", &mc_event);
        }

        std::array<Float_t, 3> pmt_pos;
        pmt_info_tree.Branch("pos", &pmt_pos);

        RAT::DU::Utility *rat_util = RAT::DU::Utility::Get();
        const RAT::DU::PMTInfo &pmt_info = rat_util->GetPMTInfo();
        RAT::DU::LightPathCalculator light_path_calculator = rat_util->GetLightPathCalculator();
        const RAT::DU::GroupVelocity &group_velocity = rat_util->GetGroupVelocity();

        std::size_t n_pmts = pmt_info.GetCount();

        for (UInt_t pmt_id = 0; pmt_id < n_pmts; pmt_id++) {
            TVector3 pmt_pos_vec = pmt_info.GetPosition(pmt_id);
            pmt_pos[0] = static_cast<Float_t>(pmt_pos_vec.X());
            pmt_pos[1] = static_cast<Float_t>(pmt_pos_vec.Y());
            pmt_pos[2] = static_cast<Float_t>(pmt_pos_vec.Z());
            pmt_info_tree.Fill();
        }

        std::size_t n_entries = dsreader.GetEntryCount();

        // Perform the cuts from the ntuple
        if (is_mc) {
            ntuple->SetBranchStatus("mcIndex", 1); 
            ntuple->SetBranchStatus("evIndex", 1); 

            if (filter != "")
                filter += " && ";
            filter = " (evIndex == 0)"; // Ignore the other triggered events
            std::cout << "Data is MC so only selecting first triggered event for every MC event\n";
        }

        ROOT::RDataFrame ntuple_df("output", ntuple_fname);

        if (filter != "") 
            std::cout << "Applying filter: " << filter << '\n';

        auto filtered_df = ntuple_df.Filter(filter);
        Long_t n_selected = filtered_df.Count().GetValue();

        if (filter != "") 
            std::cout << "Selected " << n_selected << " events out of " << n_entries << "\n";

        auto run_ids = filtered_df.Take<Int_t>("runID");
        auto event_ids = filtered_df.Take<Int_t>("eventID");

        auto *pmt_selector = RAT::PMTSelectors::PMTSelectorFactory::Get()->GetPMTSelector("PMTCalSelector");
        RAT::DS::FitVertex dummy_vertex;

        std::size_t fPSUPSystemId = RAT::DU::Point3D::GetSystemId("innerPMT");

        RAT::DU::Point3D event_pos(fPSUPSystemId);
        std::size_t n_selected_final = n_selected;
        bool valid_entry;
        std::cout << '\n';


        std::size_t run_event_i = 0; // This indexes the filtered run and event IDs
        for (std::size_t i_entry = 0; i_entry < n_entries; i_entry++) {
            if (run_event_i >= run_ids->size()) {
                break;
            }

            valid_entry = false;
            
            const RAT::DS::Entry &entry = dsreader.GetEntry(i_entry);
            run_id = entry.GetRunID();
            // In MC, some entries may not have triggered events.
            if (entry.GetEVCount() == 0) {
                if (!is_mc) {
                    throw std::runtime_error("Data is not MC and no EVs in entry " + std::to_string(i_entry));
                }
                continue;
            }
            const RAT::DS::EV &ev = entry.GetEV(0);
            event_id = ev.GetGTID();

            if (run_id != run_ids->at(run_event_i) || event_id != event_ids->at(run_event_i)) {
                continue;
            }
            std::cout << "\rProcessing entry " << run_event_i + 1 << " / " << n_selected;
            run_event_i++; // Move onto the next filtered event

            if (is_mc) {
                event_pos.SetXYZ(fPSUPSystemId, entry.GetMC().GetMCParticle(0).GetPosition());
            }

            bool mcev_exists = is_mc && entry.GetMCEVCount() > 0;
            if ((mcev_exists) > 0) {
                mc_event.global_trigger_time = static_cast<Float_t>(entry.GetMCEV(0).GetGTTime());
            }

            RAT::DS::CalPMTs const * pmts = nullptr;

            RAT::DS::MCHits const * mc_hits = nullptr;

            if (mcev_exists) 
                mc_hits = &entry.GetMCEV(0).GetMCHits();
            if (eca_cal) {
                auto types = ev.GetPartialPMTCalTypes();
                if (std::find(types.begin(), types.end(), RAT::PMTCalib::ECA) == types.end()) {
                    std::cout << "\nECA PMTCal not found in event " << i_entry + 1 << " / " << n_selected << "\n";
                } else {
                    pmts = &ev.GetPartialCalPMTs(RAT::PMTCalib::ECA);
                }
            } else {
                pmts = &ev.GetCalPMTs();
            }

            std::vector<RAT::FitterPMT> fitter_pmts;
            if (pmts != nullptr) {
                for (size_t i_pmt = 0; i_pmt < pmts->GetCount(); i_pmt++)
                    fitter_pmts.push_back(RAT::FitterPMT(pmts->GetPMT(i_pmt)));
            } // Don't fill fitter_pmts if there are no PMTs in the event.

            // Additionally select pmtData so it *only* contains PMTs which pass the PMTCal groups recommended selector cuts.
            fitter_pmts = pmt_selector->GetSelectedPMTs(fitter_pmts, dummy_vertex);

            pmt_event.id.resize(0);
            pmt_event.hit_time.resize(0);
            pmt_event.qhs.resize(0);
            if (is_mc) {
                mc_event.times_of_flight.resize(0);
            }

            for (const RAT::FitterPMT &fitter_pmt : fitter_pmts) {
                Float_t cht = static_cast<Float_t>(fitter_pmt.GetTime());
                Float_t qhs = static_cast<Float_t>(fitter_pmt.GetQHS());

                bool valid_hit = (qhs >= min_qhs) && (qhs <= max_qhs) && (cht >= min_ht) && (cht <= max_ht);

                if (valid_hit) {
                    pmt_event.id.push_back(fitter_pmt.GetID());
                    pmt_event.hit_time.push_back(cht);
                    pmt_event.qhs.push_back(qhs);
                }
                
                valid_entry = valid_entry || valid_hit;

                if (is_mc && valid_hit) {
                    RAT::DU::Point3D pmt_pos(fPSUPSystemId, pmt_info.GetPosition(fitter_pmt.GetID()));
                    light_path_calculator.CalcByPosition(event_pos, pmt_pos);
                    Double_t inner_av = light_path_calculator.GetDistInInnerAV();
                    Double_t av = light_path_calculator.GetDistInAV();
                    Double_t water = light_path_calculator.GetDistInWater();
                    Float_t time_of_flight = static_cast<Float_t>(group_velocity.CalcByDistance(inner_av, av, water));
                    
                    mc_event.times_of_flight.push_back(time_of_flight);
                }
            }
            if (valid_entry) {
                Long_t ntuple_entry_num = ntuple->GetEntryNumberWithIndex(run_id, event_id);
                if (ntuple_entry_num < 0) {
                    throw std::runtime_error("Could not find entry with runID " + std::to_string(run_id) + 
                                            " and eventID " + std::to_string(event_id));
                }
                ntuple->GetEntry(ntuple_entry_num);
                event_tree->Fill();
            }
            else
                n_selected_final--;
        }
        if (run_event_i != run_ids->size()) {
            throw std::runtime_error("Did not get through all filtered events");
        }
        std::cout << std::endl;
        std::cout << "Removed " << n_selected - n_selected_final << "\n";
        std::cout << "Writing output file " << output_fname << "\n";
        
        output_file.Write();
        output_file.Close();
        ntuple_file.Close();
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\n";
        exit(1);
    } catch (...) {
        std::cerr << "Unknown error\n";
        exit(2);
    }

}
