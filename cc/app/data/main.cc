#include <string.h>

#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/program_options.hpp>

#include "app/data/filter_extractor.h"
#include "app/data/match_extractor.h"

namespace po = boost::program_options;
namespace fs = boost::filesystem;

int main(int argc, char** argv) {
  printf("Data extractor\n");

  po::options_description desc("Allowed options");
  desc.add_options()
    ("tsf-data-dir", po::value<std::string>(), "TSF Data Directory")
    ("kitti-log-date", po::value<std::string>(), "KITTI data")
    ("log-num,l", po::value<int>(), "KITTI Log Number")
    ("save-path,p", po::value<fs::path>(), "Where to save matches")
    ;

  // Read options
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  std::string home_dir = getenv("HOME");
  std::string tsf_data_dir = home_dir + "/data/tsf_data";
  if (vm.count("tsf-data-dir")) {
    tsf_data_dir = vm["tsf-data-dir"].as<std::string>();
  } else {
    printf("Using default tsf data dir: %s\n", tsf_data_dir.c_str());
  }

  std::string kitti_log_date = "2011_09_26";
  if (vm.count("kitti-log-date")) {
    kitti_log_date = vm["kitti-log-date"].as<std::string>();
  } else {
    printf("Using default KITTI date: %s\n", kitti_log_date.c_str());
  }

  int log_num = -1;
  if (vm.count("log-num")) {
    log_num = vm["log-num"].as<int>();
  }

  fs::path save_path;
  if (vm.count("save-path")) {
    save_path = vm["save-path"].as<fs::path>();
  } else {
    printf("Need save path!\n");
    return 1;
  }

  printf("Using log %d\n", log_num);

  // Check that path exists
  std::string dir_name = (boost::format("%s_drive_%04d_sync") % kitti_log_date % log_num).str();
  fs::path base_path = fs::path(tsf_data_dir) / "kittidata" / kitti_log_date / dir_name;

  if (!fs::exists(base_path)) {
    printf("Cannot read data from %s\n", base_path.c_str());
    return 1;
  }

  if (!fs::exists(save_path)) {
    fs::create_directories(save_path);
  }

  printf("Running filter extractor\n");
  app::data::FilterExtractor fe(base_path, save_path);
  fe.Run();

  printf("Running match extractor\n");
  app::data::MatchExtractor me(base_path, save_path);
  me.Run();

  printf("\nDone\n");

  return 0;
}
