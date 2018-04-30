#include "app/flow/app.h"

#include <map>

#include <boost/format.hpp>

#include "library/kitti/util.h"
#include "library/timer/timer.h"

namespace app {
namespace flow {

App::App(const fs::path &tsf_dir, const std::string &date, int log_num, int frame_num) :
 scan_at_(frame_num),
 flow_processor_("/home/aushani/koopa_training/"),
 node_manager_(tsf_dir / "osg_models" / "lexus" / "lexus_hs.obj") {
  // Load data
  LoadVelodyneData(tsf_dir, date, log_num);
  LoadTrackletData(tsf_dir, date, log_num);
  LoadPoses(tsf_dir, date, log_num);
  printf("Done loading data\n");

  // Initialize flow processor with first scan
  flow_processor_.Initialize(scans_[scan_at_]);
  printf("Initalized Flow Processor\n");

  // Start command processing thread
  command_thread_ = std::thread(&App::ProcessCommands, this);
  printf("Started command thread\n");
}

App::App(const App &app) :
 scans_(app.scans_),
 tracklets_(app.tracklets_),
 raw_poses_(app.raw_poses_),
 sm_poses_(app.sm_poses_),
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

void App::ProcessFrame(int frame_num) {
  printf("Processing frame %d\n", frame_num);

  library::timer::Timer timer;

  const auto &scan = scans_[frame_num];

  timer.Start();
  flow_processor_.Update(scan);
  printf("Took %5.3f ms to compute flow\n", timer.GetMs());

  // Update node manager
  printf("Update node manager...\n");
  node_manager_.Update(flow_processor_, scans_[frame_num-1], scans_[frame_num], &tracklets_, frame_num);
  printf("Done\n");
}

void App::ProcessNext() {
  scan_at_++;
  ProcessFrame(scan_at_);
}

void App::QueueCommand(const Command &command) {
  command_queue_.Push(command);
}

void App::HandleClick(const Command &command) {
  double x = command.GetClickX();
  double y = command.GetClickY();

  printf("Click at %f, %f\n", x, y);

  node_manager_.ShowDistanceMap(flow_processor_, x, y);

  // Get classification result
  const auto &cm = flow_processor_.GetClassificationMap();
  const auto &probs = cm.GetClassProbabilitiesXY(x, y);

  printf("\n");
  for (const auto &kv : probs) {
    printf("%10s:\t %7.3f %%\n", kt::ObjectClassToString(kv.first).c_str(), 100.0 * kv.second);
  }
  printf("\n");

  // Get flow result
  const auto &fi = flow_processor_.GetFlowImage();
  printf("Flow is %d %d (%s)\n", fi.GetXFlowXY(x, y), fi.GetYFlowXY(x, y),
                                 fi.GetFlowValid(x, y) ? "valid":"not valid");
}

void App::HandleClearDistanceMap(const Command &command) {
  node_manager_.ClearDistanceMap();
}

void App::HandleViewMode(const Command &command) {
  int view_mode = command.GetViewMode();
  node_manager_.SetViewMode(view_mode);
}

void App::ProcessCommands() {
  while (running_) {
    const Command c = command_queue_.Pop(10);

    if (c.GetCommandType() == Type::CLEAR_DM) {
      HandleClearDistanceMap(c);
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
