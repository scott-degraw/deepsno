#include <RAT/DS/Entry.hh>
#include <RAT/DS/PMT.hh>
#include <RAT/DU/DSReader.hh>
#include <RAT/DU/Utility.hh>
#include <algorithm>
#include <filesystem>
#include <highfive/H5Easy.hpp>
#include <iostream>
#include <map>
#include <system_error>
#include <vector>

namespace fs = std::filesystem;

namespace HF = HighFive;

template <typename T> class Vector2D {
  public:
    Vector2D(std::size_t n0, std::size_t n1) : n0(n0), n1(n1) { storage = new T[n0 * n1]; }
    Vector2D(std::size_t n0, std::size_t n1, const T &fill_value) : n0(n0), n1(n1) {
        storage = new T[n0 * n1];
        std::fill(storage, storage + n0 * n1, fill_value);
    }
    ~Vector2D() { delete storage; }

    inline std::size_t size_0() const { return n0; }
    inline std::size_t size_1() const { return n1; }
    inline const T *data() const { return storage; }

    inline T &operator()(std::size_t i_0, std::size_t i_1) { return storage[i_0 * n1 + i_1]; }
    inline const T &operator()(std::size_t i_0, std::size_t i_1) const { return storage[i_0 * n1 + i_1]; }

  private:
    std::size_t n0;
    std::size_t n1;
    T *storage;
};

template <typename T> std::ostream &operator<<(std::ostream &os, const Vector2D<T> &x) {
    for (std::size_t i = 0; i < x.size_0(); i++) {
        for (std::size_t j = 0; j < x.size_1(); j++) {
            os << x(i, j) << " ";
        }
        os << "\n";
    }

    return os;
}

void test_vector() {
    Vector2D<double> x(2, 3, 0);
    x(0, 0) = 1;
    x(1, 2) = 3;
    x(0, 1) = 9;

    std::cout << x << "\n";
}

void ratds_extract(std::string input_filename, std::string output_filename, std::size_t context_window, 
        std::size_t max_triggers = 1) {
    std::cout << "Extracting data from " << input_filename << " into " << output_filename << "\n";

    fs::path path(input_filename);

    if (fs::is_directory(path)) {
        std::cerr << "Error: " << path << " is a directory." << "\n";
    } else if (!fs::exists(path)) {
        std::cerr << "Error: " << path << " not found." << "\n";
    }

    HF::File h5_file(output_filename, HF::File::ReadWrite | HF::File::Create | HF::File::Truncate);

    RAT::DU::DSReader dsreader(path.string());
    dsreader.BeginOfRun();

    auto run_info = dsreader.GetRun();
    bool is_mc = run_info.GetMCFlag();

    h5_file.createAttribute("is_mc", run_info.GetMCFlag());

    auto pmt_info_group = h5_file.createGroup("pmt_info");

    const RAT::DU::PMTInfo &pmt_info = RAT::DU::Utility::Get()->GetPMTInfo();
    std::size_t n_pmts = pmt_info.GetCount();

    auto pmt_pos_group = pmt_info_group.createGroup("position");

    // TODO: add in coordinate system
    std::vector<Float_t> pmt_x_pos(n_pmts, 0);
    std::vector<Float_t> pmt_y_pos(n_pmts, 0);
    std::vector<Float_t> pmt_z_pos(n_pmts, 0);

    for (UInt_t pmt_id = 0; pmt_id < n_pmts; pmt_id++) {
        const TVector3 pos = pmt_info.GetPosition(pmt_id);
        pmt_x_pos.at(pmt_id) = pos.X();
        pmt_y_pos.at(pmt_id) = pos.Y();
        pmt_z_pos.at(pmt_id) = pos.Z();
    }

    pmt_pos_group.createDataSet("x", pmt_x_pos);
    pmt_pos_group.createDataSet("y", pmt_y_pos);
    pmt_pos_group.createDataSet("z", pmt_z_pos);

    std::size_t n_entries = dsreader.GetEntryCount();

    std::size_t all_evs = 0;
    for (std::size_t i_entry = 0; i_entry < n_entries; i_entry++) {
        const RAT::DS::Entry &entry = dsreader.GetEntry(i_entry);
        std::size_t n_evs = std::min(max_triggers, entry.GetEVCount());
        all_evs += n_evs;
    }

    h5_file.createAttribute<std::size_t>("number_of_events", all_evs);

    std::vector<Float_t> mc_global_trigger_time;
    std::vector<Float_t> mc_event_pos_x;
    std::vector<Float_t> mc_event_pos_y;
    std::vector<Float_t> mc_event_pos_z;

    std::vector<Double_t> mc_energy;

    if (is_mc) {
        mc_global_trigger_time.resize(all_evs, -99999);
        mc_event_pos_x.resize(all_evs, 0);
        mc_event_pos_y.resize(all_evs, 0);
        mc_event_pos_z.resize(all_evs, 0);
        mc_energy.resize(all_evs, 0);
    }

    auto cal_pmt_events_group = h5_file.createGroup("cal_pmt_events");

    Vector2D<UInt_t> cal_pmt_ids(all_evs, context_window, 0); // PMT id of 0 corresponds to PMT that does not exist
    Vector2D<Float_t> cal_pmt_times(all_evs, context_window, 0);

    std::size_t evs_counter = 0;
    for (std::size_t i_entry = 0; i_entry < n_entries; i_entry++) {
        const RAT::DS::Entry &entry = dsreader.GetEntry(i_entry);
        std::size_t n_evs = std::min(max_triggers, entry.GetEVCount());
        for (std::size_t i_evs = 0; i_evs < n_evs; i_evs++) {
            if (is_mc) {
                const RAT::DS::MC &mc_event = entry.GetMC();
                const RAT::DS::MCParticle &mc_pcle = mc_event.GetMCParticle(0);
                const TVector3 pos = mc_pcle.GetPosition();
                mc_event_pos_x.at(evs_counter) = pos.X();
                mc_event_pos_y.at(evs_counter) = pos.Y();
                mc_event_pos_z.at(evs_counter) = pos.Z();
                mc_energy.at(evs_counter) = mc_pcle.GetKineticEnergy();

                if (entry.GetMCEVCount() > 0)
                    mc_global_trigger_time.at(evs_counter) = static_cast<Float_t>(entry.GetMCEV(i_evs).GetGTTime());
            }

            const RAT::DS::EV &ev = entry.GetEV(i_evs);
            const RAT::DS::CalPMTs &cal_pmts = ev.GetCalPMTs();
            std::size_t n_cal_pmts = std::min(cal_pmts.GetCount(), context_window);
            for (std::size_t i_pmt = 0; i_pmt < n_cal_pmts; i_pmt++) {
                const RAT::DS::PMTCal &cal_pmt = cal_pmts.GetPMT(i_pmt);
                cal_pmt_ids(evs_counter, i_pmt) = cal_pmt.GetID();
                Float_t pmt_time = static_cast<Float_t>(cal_pmt.GetTime());
                cal_pmt_times(evs_counter, i_pmt) = pmt_time;
            }
            evs_counter++;
        }
    }

    if (is_mc) {
        auto mc_truth_group = h5_file.createGroup("mc_truth");
        mc_truth_group.createDataSet("global_trigger_time", mc_global_trigger_time);

        auto mc_pos_group = mc_truth_group.createGroup("position");

        mc_pos_group.createDataSet("x", mc_event_pos_x);
        mc_pos_group.createDataSet("y", mc_event_pos_y);
        mc_pos_group.createDataSet("z", mc_event_pos_z);

        mc_truth_group.createDataSet("kinetic_energy", mc_energy);
    }

    HF::DataSpace cal_pmt_dataspace(cal_pmt_times.size_0(), cal_pmt_times.size_1());

    auto cal_pmt_times_dset = cal_pmt_events_group.createDataSet<Float_t>("hit_times", cal_pmt_dataspace);
    cal_pmt_times_dset.write_raw(cal_pmt_times.data());
    auto cal_pmt_ids_dset = cal_pmt_events_group.createDataSet<UInt_t>("ids", cal_pmt_dataspace);
    cal_pmt_ids_dset.write_raw(cal_pmt_ids.data());
}