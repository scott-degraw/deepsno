#include <RAT/DS/Entry.hh>
#include <RAT/DS/PMT.hh>
#include <RAT/DU/DSReader.hh>
#include <RAT/DU/Utility.hh>
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


constexpr Float_t kFloatNaN = std::numeric_limits<Float_t>::quiet_NaN();
constexpr Double_t kDoubleNaN = std::numeric_limits<Double_t>::quiet_NaN();

struct Event {
    std::vector<Float_t> av_offset;

    Event() : av_offset(3) {}
};

struct CalEvent {
    std::vector<UInt_t> pmt_ids;
    std::vector<Float_t> hit_times;
    std::vector<Float_t> qhs;
};

struct PmtInfo {
    std::vector<Float_t> pmt_pos;
};

struct McEvent {
    std::vector<Float_t> times_of_flight;
    std::vector<Float_t> hit_times;
    std::vector<Float_t> event_pos;
    Float_t global_trigger_time = kFloatNaN;
    Float_t kinetic_energy = kFloatNaN;
};

struct EcaEvent {
    std::vector<Float_t> hit_times;
};


void ratds_extract(std::string input_filename, std::string output_filename,
        Float_t min_ht, Float_t max_ht, Float_t min_qhs, Float_t max_qhs,
        std::string filter = "", bool eca_cal = false) {
    std::cout << "Extracting data from " << input_filename << " into " << output_filename << "\n";

    if (gSystem->AccessPathName(input_filename.c_str(), kFileExists) != 0) {
        throw std::runtime_error("Input file " + input_filename + " does not exist");
    } 

    RAT::DU::DSReader dsreader(input_filename);
    dsreader.BeginOfRun();

    auto run_info = dsreader.GetRun();
    bool is_mc = run_info.GetMCFlag();

    RAT::DB *db = RAT::DB::Get();

    TFile output_file(output_filename.c_str(), "RECREATE");
    TTree pmt_info_tree("pmt_info", "Contains PMT information");
    TTree event_tree("events", "Contains event level data");

    std::vector<Float_t> av_offset_vec = db->GetLink("GEO", "av")->GetFArrayFromD("position");
    RAT::DBLinkPtr native_geo_dims_link = db->GetLink("NATIVE_GEO_DIMENSIONS", "natgeo_dimensions");
    Double_t inner_av_radius = native_geo_dims_link->GetD("inner_av_radius");
    Float_t av_thickness = native_geo_dims_link->GetD("av_thickness");

    TParameter<Float_t> param_inner_av_radius("inner_av_radius", static_cast<Float_t>(inner_av_radius));
    TParameter<Float_t> param_av_thickness("av_thickness", static_cast<Float_t>(av_thickness));
    param_inner_av_radius.Write();
    param_av_thickness.Write();

    Event event;
    CalEvent cal_event;
    EcaEvent eca_event;
    McEvent mc_event;

    event_tree.Branch("event", &event);
    event.av_offset = av_offset_vec;

    event_tree.Branch("cal", &cal_event);
    if (eca_cal) {
        event_tree.Branch("eca", &eca_event);
    }
    if (is_mc) {
        event_tree.Branch("mc", &mc_event);
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

        std::string ntuple_fname(input_filename);

        unsigned last_dot = ntuple_fname.find_last_of('.');
        ntuple_fname.erase(ntuple_fname.begin() + last_dot, ntuple_fname.end());
        ntuple_fname += ".ntuple.root";

        if (gSystem->AccessPathName(ntuple_fname.c_str(), kFileExists) != 0) {
            throw std::runtime_error("A filter was given but there is no corresponding ntuple to " + input_filename);
        }
        TFile* ntuple_file = TFile::Open(ntuple_fname.c_str(), "READ");
        TTree* ntuple = (TTree*)ntuple_file->Get("output;1");   

        ntuple->Draw(">>entry_list", filter.c_str(), "entrylist");
        TEntryList * entry_list = (TEntryList*) gDirectory->Get("entry_list");

        n_selected = entry_list->GetN();
        std::cout << "Selected " << n_selected << " events out of " << n_entries << "\n";

        entry_indices.resize(n_selected);
        for (std::size_t i = 0; i < n_selected; i++) {
            entry_indices[i] = entry_list->Next();
        }

        ntuple_file->Close();
    } else {
        entry_indices.resize(n_entries);
        std::cout << n_entries << " entries in the dataset\n";
        std::iota(entry_indices.begin(), entry_indices.end(), 0);
        std::cout << entry_indices[0] << " " << entry_indices[n_entries-1] << "\n";
    }


    std::size_t fPSUPSystemId = RAT::DU::Point3D::GetSystemId("innerPMT");

    RAT::DU::Point3D event_pos(fPSUPSystemId);
    std::size_t n_selected_final = n_selected;
    for (std::size_t i_select_entry = 0; i_select_entry < entry_indices.size(); i_select_entry++) {
        bool valid_entry = false;
        std::cout << "\rProcessing entry " << i_select_entry + 1 << " / " << n_selected << std::flush;
        const RAT::DS::Entry &entry = dsreader.GetEntry(entry_indices[i_select_entry]);
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
        const RAT::DS::CalPMTs &cal_pmts = ev.GetCalPMTs();
        RAT::DS::MCHits const * mc_hits = nullptr;
        RAT::DS::CalPMTs const * partial_cal_pmts = nullptr;

        if (is_mc) 
            mc_hits = &entry.GetMCEV(0).GetMCHits();
        if (eca_cal) {
            // 1 corresponds to the ECA calibrated PMTs
            partial_cal_pmts = &ev.GetPartialCalPMTs(1);
        }

        std::size_t n_cal_pmts = cal_pmts.GetCount();
        cal_event.pmt_ids.resize(0);
        cal_event.hit_times.resize(0);
        cal_event.qhs.resize(0);
        if (eca_cal) 
            eca_event.hit_times.resize(0);
        if (is_mc)
            mc_event.times_of_flight.resize(0);
            mc_event.hit_times.resize(0);
        for (std::size_t i_pmt = 0; i_pmt < n_cal_pmts; i_pmt++) {
            const RAT::DS::PMTCal &cal_pmt = cal_pmts.GetPMT(i_pmt);
            Float_t cht = static_cast<Float_t>(cal_pmt.GetTime());
            Float_t qhs = static_cast<Float_t>(cal_pmt.GetQHS());

            bool valid_hit = (qhs >= min_qhs) && (qhs <= max_qhs);

            if (eca_cal) {
                const RAT::DS::PMTCal &eca_cal_pmt = partial_cal_pmts->GetPMT(i_pmt);
                Float_t eca_hit_time = static_cast<Float_t>(eca_cal_pmt.GetTime());

                valid_hit = valid_hit && (eca_hit_time >= min_ht) && (eca_hit_time <= max_ht);
                if (valid_hit)
                    eca_event.hit_times.push_back(eca_hit_time);
            } else {
                valid_hit = valid_hit && (cht >= min_ht) && (cht <= max_ht);
            }

            if (valid_hit) {
                cal_event.pmt_ids.push_back(cal_pmt.GetID());
                cal_event.hit_times.push_back(cht);
                cal_event.qhs.push_back(qhs);
            }
            
            valid_entry = valid_entry || valid_hit;

            if (is_mc && valid_hit) {
                RAT::DU::Point3D pmt_pos(fPSUPSystemId, pmt_info.GetPosition(cal_pmt.GetID()));
                light_path_calculator.CalcByPosition(event_pos, pmt_pos);
                Double_t inner_av = light_path_calculator.GetDistInInnerAV();
                Double_t av = light_path_calculator.GetDistInAV();
                Double_t water = light_path_calculator.GetDistInWater();
                Float_t time_of_flight = static_cast<Float_t>(group_velocity.CalcByDistance(inner_av, av, water));
                
                const RAT::DS::MCHit &mc_hit = mc_hits->GetPMT(i_pmt);
                mc_event.hit_times.push_back(static_cast<Float_t>(mc_hit.GetTime()));
                mc_event.times_of_flight.push_back(time_of_flight);
            }
        }
        if (valid_entry)
            event_tree.Fill();
        else
            n_selected_final--;
    }
    std::cout << std::endl;
    std::cout << "Removed " << n_selected - n_selected_final << " events due to bad PMT hits\n";
    std::cout << "Writing output file " << output_filename << "\n";
    
    output_file.Write();
    output_file.Close();
}