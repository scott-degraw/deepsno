#include <string>
#include <fstream>
#include <ROOT/RDataFrame.hxx>
#include <TSystem.h>

void ntuple_2_eventlist(std::string ntuple_filename, std::string filter) {
    try {
        std::string output_filename = "eventlist.txt";

        if (gSystem->AccessPathName(ntuple_filename.c_str(), kFileExists) != 0) {
            throw std::runtime_error("The ntuple file " + ntuple_filename + " does not exist.");
        }

        ROOT::RDataFrame df("output", ntuple_filename);
        std::cout << "Applying filter: " << filter << "\n";

        auto filtered_df = df.Filter(filter);

        std::ofstream out_file(output_filename);
        if (!out_file.is_open()) {
            throw std::runtime_error("Could not open output file " + output_filename);
        }

        filtered_df.Foreach(
            [&](Int_t run_id, Int_t event_id) {
                out_file << run_id << "," << event_id << "\n";
            },
            {"runID", "eventID"}
        );
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\n";
        exit(1);
    } catch (...) {
        std::cerr << "Unknown error\n";
        exit(2);
    }
}