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

void ratds_extract(std::string input_filename, std::string output_filename, float min_hit_time, float max_hit_time) {
    std::cout << "Extracting data from " << input_filename << " into " << output_filename << "\n";
    std::size_t max_triggers = 1; // Make this into an argument

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
    std::size_t n_pmts_all_types = pmt_info.GetCount();

    // TODO: Maybe make this a map
    std::vector<unsigned> pmt_id_2_index(n_pmts_all_types, 0);

    std::vector<unsigned> inward_pmt_ids;

    unsigned pmt_index = 0;
    for (unsigned pmt_id = 0; pmt_id < n_pmts_all_types; pmt_id++) {
        RAT::DU::PMTInfo::EPMTType pmt_type = pmt_info.GetType(pmt_id);
        // Check if the pmt is an inward PMT
        if (pmt_type == RAT::DU::PMTInfo::EPMTType::NORMAL || pmt_type == RAT::DU::PMTInfo::EPMTType::HQE) {
            inward_pmt_ids.push_back(pmt_id);
            pmt_id_2_index.at(pmt_id) = pmt_index;
            pmt_index++;
        }
    }
    unsigned n_inward_pmts = pmt_index;

    pmt_info_group.createAttribute<unsigned>("n_inward_pmts", n_inward_pmts);

    auto pmt_id_2_index_dataset = pmt_info_group.createDataSet("pmt_id_2_index", pmt_id_2_index);
    auto inward_pmt_ids_dataset = pmt_info_group.createDataSet("inward_pmt_ids", inward_pmt_ids);

    auto pmt_pos_group = pmt_info_group.createGroup("position");

    // TODO: add in coordinate system
    std::vector<float> pmt_x_pos(n_inward_pmts, 0);
    std::vector<float> pmt_y_pos(n_inward_pmts, 0);
    std::vector<float> pmt_z_pos(n_inward_pmts, 0);

    for (unsigned pmt_index = 0; pmt_index < n_inward_pmts; pmt_index++) {
        unsigned pmt_id = inward_pmt_ids[pmt_index];
        const TVector3 pos = pmt_info.GetPosition(pmt_id);
        pmt_x_pos.at(pmt_index) = pos.X();
        pmt_y_pos.at(pmt_index) = pos.Y();
        pmt_z_pos.at(pmt_index) = pos.Z();
    }

    auto x_pos_dset = pmt_pos_group.createDataSet("x", pmt_x_pos);
    auto y_pos_dset = pmt_pos_group.createDataSet("y", pmt_y_pos);
    auto z_pos_dset = pmt_pos_group.createDataSet("z", pmt_z_pos);

    std::size_t n_entries = dsreader.GetEntryCount();

    std::size_t all_evs = 0;
    for (std::size_t i_entry = 0; i_entry < n_entries; i_entry++) {
        const RAT::DS::Entry &entry = dsreader.GetEntry(i_entry);
        std::size_t n_evs = std::min(max_triggers, entry.GetEVCount());
        all_evs += n_evs;
    }

    h5_file.createAttribute<std::size_t>("number_of_events", all_evs);

    std::vector<float> mc_event_pos_x;
    std::vector<float> mc_event_pos_y;
    std::vector<float> mc_event_pos_z;

    if (is_mc) {
        mc_event_pos_x.resize(all_evs, 0);
        mc_event_pos_y.resize(all_evs, 0);
        mc_event_pos_z.resize(all_evs, 0);
    }

    auto cal_pmt_events_group = h5_file.createGroup("cal_pmt_events");

    Vector2D<float> cal_pmt_times(all_evs, n_inward_pmts, 0);
    // true indicates that there the PMT was not hit during that event
    Vector2D<bool> cal_pmt_masks(cal_pmt_times.size_0(), cal_pmt_times.size_1(), true);

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
            }

            const RAT::DS::EV &ev = entry.GetEV(i_evs);
            const RAT::DS::CalPMTs &cal_pmts = ev.GetCalPMTs();
            std::size_t n_cal_pmts = cal_pmts.GetCount();
            for (std::size_t i_pmt = 0; i_pmt < n_cal_pmts; i_pmt++) {
                const RAT::DS::PMTCal &cal_pmt = cal_pmts.GetPMT(i_pmt);
                unsigned pmt_id = cal_pmt.GetID();
                unsigned pmt_index = pmt_id_2_index.at(pmt_id);
                cal_pmt_masks(evs_counter, pmt_index) = false;
                float pmt_time = static_cast<float>(cal_pmt.GetTime());
                pmt_time = std::clamp(pmt_time, min_hit_time, max_hit_time);
                cal_pmt_times(evs_counter, pmt_index) = pmt_time;
            }
            evs_counter++;
        }
    }

    if (is_mc) {
        auto mc_truth_group = h5_file.createGroup("mc_truth");
        auto mc_pos_group = mc_truth_group.createGroup("position");

        mc_pos_group.createDataSet("x", mc_event_pos_x);
        mc_pos_group.createDataSet("y", mc_event_pos_y);
        mc_pos_group.createDataSet("z", mc_event_pos_z);
    }

    HF::DataSpace cal_pmt_dataspace(cal_pmt_times.size_0(), cal_pmt_times.size_1());

    auto cal_pmt_times_dset = cal_pmt_events_group.createDataSet<float>("hit_times", cal_pmt_dataspace);
    cal_pmt_times_dset.write_raw(cal_pmt_times.data());
    auto cal_pmt_masks_dset = cal_pmt_events_group.createDataSet<bool>("masks", cal_pmt_dataspace);
    cal_pmt_masks_dset.write_raw(cal_pmt_masks.data());
}