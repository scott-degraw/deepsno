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
#include <limits>
#include <memory>
#include <TEntryList.h>
#include <TFile.h>
#include <TCut.h>
#include <TSystem.h>
#include <TParameter.h>
#include <TVector3.h>

std::vector<std::string> ntuple_branches = {
    "runID", "eventID", "nhits", "fitValid", "posx", "posy", "posz", "posz_av", "energy", "time", 
    "nhitsCleaned", "nearAV", "itr", "necknhits", 
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
    std::vector<Float_t> event_pos;
    Float_t global_trigger_time = kFloatNaN;
    Float_t kinetic_energy = kFloatNaN;
};


void ratds_extract(std::string input_fname, 
        std::string ntuple_fname,
        std::string output_fname,
        Float_t min_ht, Float_t max_ht, Float_t min_qhs, Float_t max_qhs, 
        std::string filter = "", bool eca_cal = false) {
    std::cout << "Extracting data from " << input_fname << " into " << output_fname << "\n";

    if (gSystem->AccessPathName(input_fname.c_str(), kFileExists) != 0) 
        throw std::runtime_error("Input file " + input_fname + " does not exist");
    if (gSystem->AccessPathName(ntuple_fname.c_str(), kFileExists) != 0) 
        throw std::runtime_error("Ntuple file " + ntuple_fname + " does not exist");

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
    TTree * event_tree = ntuple->CloneTree(0);
    event_tree->SetDirectory(&output_file);
    event_tree->SetTitle("Contains event level data");
    event_tree->SetName("event");

    Int_t run_id;
    Int_t event_id;
    ntuple->BuildIndex("runID", "eventID");

    std::vector<Float_t> av_offset_vec = db->GetLink("GEO", "av")->GetFArrayFromD("position");
    Float_t av_offset[3];
    for (size_t i = 0; i < av_offset_vec.size(); i++) {
        av_offset[i] = av_offset_vec[i];
    }
    event_tree->Branch("av_offset", av_offset, "av_offset[3]/F");

    RAT::DBLinkPtr native_geo_dims_link = db->GetLink("NATIVE_GEO_DIMENSIONS", "natgeo_dimensions");
    Double_t inner_av_radius = native_geo_dims_link->GetD("inner_av_radius");
    Float_t av_thickness = native_geo_dims_link->GetD("av_thickness");

    TParameter<Float_t> param_inner_av_radius("inner_av_radius", static_cast<Float_t>(inner_av_radius));
    TParameter<Float_t> param_av_thickness("av_thickness", static_cast<Float_t>(av_thickness));
    param_inner_av_radius.Write();
    param_av_thickness.Write();

    PmtEvent pmt_event;
    McEvent mc_event;
    mc_event.event_pos.resize(3);

    event_tree->Branch("pmt_id", &pmt_event.id);
    event_tree->Branch("pmt_hit_time", &pmt_event.hit_time);
    event_tree->Branch("pmt_qhs", &pmt_event.qhs);
    if (is_mc) {
        event_tree->Branch("mc", &mc_event);
    }

    std::vector<Float_t> pmt_pos(3);
    pmt_info_tree.Branch("pos", &pmt_pos);

    RAT::DU::Utility *rat_util = RAT::DU::Utility::Get();
    const RAT::DU::PMTInfo &pmt_info = rat_util->GetPMTInfo();
    RAT::DU::LightPathCalculator light_path_calculator = rat_util->GetLightPathCalculator();
    const RAT::DU::GroupVelocity &group_velocity = rat_util->GetGroupVelocity();

    std::size_t n_pmts = pmt_info.GetCount();

    for (UInt_t pmt_id = 0; pmt_id < n_pmts; pmt_id++) {
        const TVector3 pos = pmt_info.GetPosition(pmt_id);
        pos.GetXYZ(pmt_pos.data());
        pmt_info_tree.Fill();
    }

    std::size_t n_entries = dsreader.GetEntryCount();

    // Perform the cuts from the ntuple
    std::size_t n_selected = n_entries;
    std::vector<std::size_t> entry_indices;

    if (filter != "") {
        std::cout << "Applying filter: " << filter << '\n';

        if (is_mc) 
            ntuple->SetBranchStatus("mcIndex", 1); 
            ntuple->SetBranchStatus("evIndex", 1); 
            filter += " && (evIndex == 0)"; // Ignore the other triggered events
            std::cout << "Data is MC so only selecting first triggered event for every MC event\n";
        ntuple->Draw(">>entry_list", filter.c_str(), "entrylist");
        TEntryList * entry_list = (TEntryList*) gDirectory->Get("entry_list");

        n_selected = entry_list->GetN();
        std::cout << "Selected " << n_selected << " events out of " << n_entries << "\n";

        entry_indices.resize(n_selected);
        for (std::size_t i = 0; i < n_selected; i++) {
            entry_indices[i] = entry_list->Next();
        }
    
        if (is_mc) {
            Int_t mc_index;
            ntuple->SetBranchAddress("mcIndex", &mc_index);
            for (std::size_t i = 0; i < entry_indices.size(); i++) {
                ntuple->GetEntry(entry_indices[i]);
                entry_indices[i] = mc_index;
            }
        }

    } else {
        entry_indices.resize(n_entries);
        std::cout << n_entries << " entries in the dataset\n";
        std::iota(entry_indices.begin(), entry_indices.end(), 0);
    }

    auto *pmt_selector = RAT::PMTSelectors::PMTSelectorFactory::Get()->GetPMTSelector("PMTCalSelector");
    RAT::DS::FitVertex dummy_vertex;

    std::size_t fPSUPSystemId = RAT::DU::Point3D::GetSystemId("innerPMT");

    RAT::DU::Point3D event_pos(fPSUPSystemId);
    std::size_t n_selected_final = n_selected;
    bool valid_entry;
    for (std::size_t i_select_entry = 0; i_select_entry < entry_indices.size(); i_select_entry++) {
        valid_entry = false;
        std::cout << "Processing entry " << i_select_entry + 1 << " / " << n_selected << std::endl;
        
        std::size_t entry_index = entry_indices[i_select_entry];
        assert((entry_index < n_entries) && "Trying to access entry with index that doesn't exist");
        const RAT::DS::Entry &entry = dsreader.GetEntry(entry_index);
        run_id = entry.GetRunID();
        // In MC, some entries may not have triggered events.
        if (entry.GetEVCount() == 0) {
            if (!is_mc) {
                throw std::runtime_error("Data is not MC and no EVs in entry " + std::to_string(i_select_entry));
            }
            continue;
        }
        if (is_mc) {
            const RAT::DS::MC &mc_entry = entry.GetMC();
            const RAT::DS::MCParticle &mc_pcle = mc_entry.GetMCParticle(0);
            event_pos.SetXYZ(fPSUPSystemId, mc_pcle.GetPosition());
            mc_event.event_pos.at(0) = event_pos.X();
            mc_event.event_pos.at(1) = event_pos.Y();
            mc_event.event_pos.at(2) = event_pos.Z();
            mc_event.kinetic_energy = mc_pcle.GetKineticEnergy();

            if (entry.GetMCEVCount() > 0)
                mc_event.global_trigger_time = static_cast<Float_t>(entry.GetMCEV(0).GetGTTime());
        }

        const RAT::DS::EV &ev = entry.GetEV(0);
        event_id = ev.GetGTID();

        RAT::DS::CalPMTs const * pmts = nullptr;

        RAT::DS::MCHits const * mc_hits = nullptr;

        if (is_mc) 
            mc_hits = &entry.GetMCEV(0).GetMCHits();
        if (eca_cal) {
            auto types = ev.GetPartialPMTCalTypes();
            if (std::find(types.begin(), types.end(), RAT::PMTCalib::ECA) == types.end()) {
                std::cout << "\nECA PMTCal not found in event " << i_select_entry + 1 << " / " << n_selected << "\n";
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
    std::cout << std::endl;
    std::cout << "Removed " << n_selected - n_selected_final << "\n";
    std::cout << "Writing output file " << output_fname << "\n";
    
    output_file.Write();
    output_file.Close();
    ntuple_file.Close();
}
