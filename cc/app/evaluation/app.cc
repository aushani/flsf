#include "app/evaluation/app.h"

#include <boost/format.hpp>

#include "library/evaluation/flow_image_evaluator.h"
#include "library/timer/timer.h"

namespace ev = library::evaluation;

namespace app {
namespace evaluation {

App::App(const fs::path &tsf_dir, const std::string &date, int log_num) :
 camera_cal_(tsf_dir / "kittidata" / date),
 flow_processor_("/home/aushani/koopa_training/") {
  // Load data
  LoadVelodyneData(tsf_dir, date, log_num);
  LoadTrackletData(tsf_dir, date, log_num);
  LoadPoses(tsf_dir, date, log_num);
  printf("Done loading data\n");

  // Initialize flow processor with first scan
  flow_processor_.Initialize(scans_[0]);
  printf("Initalized Flow Processor\n");
}

void App::LoadVelodyneData(const fs::path &tsf_dir, const std::string &date, int log_num) {
  library::timer::Timer t;

  printf("Loading velodyne data...\n");

  std::string dir_name = (boost::format("%s_drive_%04d_sync") % date % log_num).str();
  fs::path path = tsf_dir / "kittidata" / date / dir_name / "velodyne_points" / "data";

  scans_ = kt::VelodyneScan::LoadDirectory(path);

  printf("Took %5.3f ms to load all velodyne files\n", t.GetMs());
}

void App::LoadTrackletData(const fs::path &tsf_dir, const std::string &date, int log_num) {
  printf("Loading tracklet data...\n");

  std::string dir_name = (boost::format("%s_drive_%04d_sync") % date % log_num).str();

  fs::path tracklet_file = tsf_dir / "kittidata" / date / dir_name / "tracklet_labels.xml";
  tracklets_.loadFromFile(tracklet_file.string());
}

void App::LoadPoses(const fs::path &tsf_dir, const std::string &date, int log_num) {
  printf("Loading poses\n");

  std::string dir_name = (boost::format("%s_drive_%04d_sync") % date % log_num).str();
  fs::path path = tsf_dir / "kittidata" / date / dir_name;

  raw_poses_ = kt::Pose::LoadRawPoses(path);
  sm_poses_ = kt::Pose::LoadScanMatchedPoses(path);
}

void App::Run(const fs::path &save_path) {
  library::timer::Timer timer;

  ev::FlowImageEvaluator eval(tracklets_, camera_cal_, sm_poses_);

  for (size_t scan_at=1; scan_at<scans_.size(); scan_at++) {
    printf("Processing frame % 3ld / % 3ld\n", scan_at, scans_.size());

    const auto &scan = scans_[scan_at];

    timer.Start();
    flow_processor_.Update(scan, false); // don't need extra data
    printf("Took %5.3f ms to compute flow\n", timer.GetMs());

    // Evaluation
    eval.Evaluate(flow_processor_.GetFlowImage(), scan_at - 1, scan_at);

    printf("\n\n");
  }

  eval.WriteErrors(save_path);
}

} // evaluation
} // app
