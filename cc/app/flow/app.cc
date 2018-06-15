#include "app/flow/app.h"

#include <map>

#include <boost/format.hpp>
#include <Eigen/Core>

#include "library/kitti/util.h"
#include "library/timer/timer.h"

namespace app {
namespace flow {

App::App(const fs::path &tsf_dir, const std::string &date, const fs::path &network_path, int log_num, int frame_num) :
 camera_cal_(tsf_dir / "kittidata" / date),
 scan_at_(frame_num),
 flow_processor_(network_path),
 node_manager_(tsf_dir / "osg_models" / "lexus" / "lexus_hs.obj") {
  // Load data
  LoadVelodyneData(tsf_dir, date, log_num);
  LoadTrackletData(tsf_dir, date, log_num);
  LoadPoses(tsf_dir, date, log_num);
  printf("Done loading data\n");

  // Initialize flow processor with first scan
  flow_processor_.Initialize(scans_[scan_at_]);
  printf("Initalized Flow Processor\n");

  // Start up evaluation
  fi_eval_ = std::make_unique<ev::FlowImageEvaluator>(tracklets_, camera_cal_, sm_poses_);

  // Start command processing thread
  command_thread_ = std::thread(&App::ProcessCommands, this);
  printf("Started command thread\n");
}

App::App(const App &app) :
 scans_(app.scans_),
 tracklets_(app.tracklets_),
 raw_poses_(app.raw_poses_),
 sm_poses_(app.sm_poses_),
 camera_cal_(app.camera_cal_),
 scan_at_(app.scan_at_),
 flow_processor_(app.flow_processor_),
 node_manager_(app.node_manager_) {
  // Start command processing thread
  command_thread_ = std::thread(&App::ProcessCommands, this);
}

App::~App() {
  running_ = false;
  command_thread_.join();
}

App App::operator=(const App& app) {
  // Stop old thread
  running_ = false;
  command_thread_.join();

  running_ = true;

  scans_ = app.scans_;
  tracklets_ = app.tracklets_;
  raw_poses_ = app.raw_poses_;
  sm_poses_ = app.sm_poses_;
  camera_cal_ = app.camera_cal_;
  scan_at_ = app.scan_at_;
  flow_processor_ = app.flow_processor_;
  node_manager_ = app.node_manager_;

  // Start command processing thread
  command_thread_ = std::thread(&App::ProcessCommands, *this);

  return *this;
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

void App::SetViewer(const std::shared_ptr<vw::Viewer> &viewer) {
  node_manager_.SetViewer(viewer);
}

void App::Process() {
  printf("Processing frame %ld\n", scan_at_);

  library::timer::Timer timer;

  const auto &scan = scans_[scan_at_];

  timer.Start();
  flow_processor_.Update(scan);
  printf("Took %5.3f ms to compute flow\n", timer.GetMs());

  printf("\n\n");

  // Update node manager
  printf("Update node manager...\n");
  node_manager_.Update(flow_processor_, scans_[scan_at_-1], scans_[scan_at_], &tracklets_, scan_at_);
  printf("Done\n");

  // Evaluation
  fi_eval_->Clear();
  fi_eval_->Evaluate(flow_processor_.GetFlowImage(), flow_processor_.GetBackgroundFilterMap1(), scan_at_ - 1, scan_at_);

  printf("\n\n");
}

void App::Refresh() {
  printf("Processing frame %ld\n", scan_at_);

  library::timer::Timer timer;

  timer.Start();
  flow_processor_.Refresh();
  printf("Took %5.3f ms to refresh flow\n", timer.GetMs());

  printf("\n\n");

  // Update node manager
  printf("Update node manager...\n");
  node_manager_.Update(flow_processor_, scans_[scan_at_-1], scans_[scan_at_], &tracklets_, scan_at_);
  printf("Done\n");

  // Evaluation
  fi_eval_->Clear();
  fi_eval_->Evaluate(flow_processor_.GetFlowImage(), flow_processor_.GetBackgroundFilterMap1(), scan_at_ - 1, scan_at_);

  printf("\n\n");
}

void App::ProcessNext() {
  scan_at_++;
  Process();
}

void App::QueueCommand(const Command &command) {
  command_queue_.Push(command);
}

void App::HandleClick(const Command &command) {
  printf("\n");

  double x = command.GetClickX();
  double y = command.GetClickY();
  Eigen::Vector2f pos1(x, y);

  const auto &fm = flow_processor_.GetBackgroundFilterMap1();
  const auto &fi = flow_processor_.GetFlowImage();
  const auto &dm = flow_processor_.GetDistanceMap();
  double res = fi.GetResolution();

  int i = std::round(x / res);
  int j = std::round(y / res);

  printf("Click at %f, %f\n", x, y);
  printf("Click at %d, %d\n", i, j);

  if (camera_cal_.InCameraView(x, y, 0)) {
    printf("In camera view\n");
  } else {
    printf("NOT In camera view\n");
  }

  if (!fi.InRangeXY(x, y)) {
    printf("Out of range\n");
    return;
  } else {
    printf("In range\n");
  }

  const auto &scan2 = scans_[scan_at_];
  node_manager_.ShowDistanceMap(flow_processor_, scan2, x, y);

  // Get filter result
  float prob = fm.GetFilterProbabilityXY(x, y);

  printf("p_background: %5.3f %%\n", prob * 100.0);

  kt::ObjectClass oc = kt::GetObjectTypeAtLocation(&tracklets_, pos1, scan_at_ - 1, res);
  printf("Object is %s\n", kt::ObjectClassToString(oc).c_str());

  // Get flow result
  printf("Flow is %f %f (%s)\n", fi.GetXFlowXY(x, y)*res, fi.GetYFlowXY(x, y)*res,
                                 fi.GetFlowValidXY(x, y) ? "valid":"not valid");

  // Get ground truth
  const kt::Pose pose1 = sm_poses_[scan_at_ - 1];
  const kt::Pose pose2 = sm_poses_[scan_at_];
  bool track_disappears = false;
  Eigen::Vector2f pos2 = kt::FindCorrespondingPosition(&tracklets_, pos1, scan_at_ - 1, scan_at_, pose1, pose2, &track_disappears, res);
  Eigen::Vector2f flow = pos2 - pos1;

  if (track_disappears) {
    printf("Track disappears!\n");
  }

  printf("True flow is %f %f\n", flow.x(), flow.y());

  printf("Distance at true flow is      %f\n", dm.GetDistanceXY(x, y, flow.x(), flow.y()));
  printf("Distance at estimated flow is %f\n", dm.GetDistanceXY(x, y, fi.GetXFlowXY(x, y)*res, fi.GetYFlowXY(x, y)*res));
}

void App::HandleClearDistanceMap(const Command &command) {
  node_manager_.ClearDistanceMap();
}

void App::HandleViewMode(const Command &command) {
  int view_mode = command.GetViewMode();
  node_manager_.SetViewMode(view_mode);
}

void App::HandleIterations(const Command &command) {
  int iterations = flow_processor_.GetIterations();

  if (command.GetCommandType() == Type::INCREASE_ITERATIONS) {
    iterations++;
  }

  if (command.GetCommandType() == Type::DECREASE_ITERATIONS) {
    iterations--;
  }

  if (iterations < 1) {
    iterations = 1;
  }

  flow_processor_.SetIterations(iterations);

  printf("Set iterations to %d\n", iterations);
}

void App::HandleRefresh(const Command &command) {
  Refresh();
}

void App::ProcessCommands() {
  while (running_) {
    const Command c = command_queue_.Pop(10);

    if (c.GetCommandType() == Type::CLEAR_DM) {
      HandleClearDistanceMap(c);
    }

    if (c.GetCommandType() == Type::INCREASE_ITERATIONS) {
      HandleIterations(c);
    }

    if (c.GetCommandType() == Type::DECREASE_ITERATIONS) {
      HandleIterations(c);
    }

    if (c.GetCommandType() == Type::REFRESH) {
      HandleRefresh(c);
    }

    if (c.GetCommandType() == Type::NEXT) {
      ProcessNext();
    }

    if (c.GetCommandType() == Type::VIEW_MODE) {
      HandleViewMode(c);
    }

    if (c.GetCommandType() == Type::CLICK_AT) {
      HandleClick(c);
    }
  }
}

} // flow
} // app
