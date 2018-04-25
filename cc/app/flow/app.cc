#include "app/flow/app.h"

#include <map>

#include <boost/format.hpp>

#include "library/kitti/util.h"
#include "library/kitti/nodes/point_cloud.h"
#include "library/kitti/nodes/tracklets.h"
#include "library/flow/nodes/flow_image.h"
#include "library/flow/nodes/classification_map.h"
#include "library/osg_nodes/car.h"
#include "library/ray_tracing/nodes/occ_grid.h"
#include "library/timer/timer.h"

namespace osgn = library::osg_nodes;

namespace app {
namespace flow {

App::App(const fs::path &tsf_dir, const std::string &date, int log_num) :
 flow_processor_("/home/aushani/koopa_training/") {
  // Path for car model
  car_path_ = tsf_dir / "osg_models" / "lexus" / "lexus_hs.obj";

  // Load data
  LoadVelodyneData(tsf_dir, date, log_num);
  LoadTrackletData(tsf_dir, date, log_num);
  LoadPoses(tsf_dir, date, log_num);
  printf("Done loading data\n");

  // Initialize flow processor with first scan
  flow_processor_.Initialize(scans_[0]);
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
 viewer_(app.viewer_) {
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
  viewer_ = app.viewer_;

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
  viewer_ = viewer;
}

void App::ProcessFrame(int frame_num) {
  printf("Processing frame %d\n", frame_num);

  library::timer::Timer timer;

  const auto &scan = scans_[frame_num];

  timer.Start();
  flow_processor_.Update(scan);
  printf("Took %5.3f ms to compute flow\n", timer.GetMs());

  if (viewer_) {
    printf("Update viewer\n");
    prev_dm_ = boost::none;

    UpdateViewer();
    printf("Done\n");
  }
}

void App::ProcessNext() {
  scan_at_++;
  ProcessFrame(scan_at_);
}

void App::QueueCommand(const Command &command) {
  command_queue_.Push(command);
}

void App::HandleClick(const Command &command) {
  fl::DistanceMap dm = flow_processor_.GetDistanceMap();
  osg::ref_ptr<fl::nodes::DistanceMap> dmn = new fl::nodes::DistanceMap(dm, command.GetClickX(), command.GetClickY());

  // Remove prev dm
  if (prev_dm_) {
    viewer_->RemoveChild(*prev_dm_);
  }

  viewer_->AddChild(dmn);
  prev_dm_ = dmn;
}

void App::HandleViewMode(const Command &command) {
  view_mode_ = command.GetViewMode();
  UpdateViewer();
}

void App::UpdateViewer() {
  kt::VelodyneScan scan = scans_[scan_at_];

  rt::OccGrid og1 = flow_processor_.GetLastOccGrid1();
  rt::OccGrid og2 = flow_processor_.GetLastOccGrid2();

  fl::FlowImage fi = flow_processor_.GetFlowImage();
  fl::ClassificationMap cm = flow_processor_.GetClassificationMap();

  osg::ref_ptr<kt::nodes::PointCloud> pc = new kt::nodes::PointCloud(scan);
  osg::ref_ptr<kt::nodes::Tracklets> tn1 = new kt::nodes::Tracklets(&tracklets_, scan_at_-1);
  osg::ref_ptr<kt::nodes::Tracklets> tn2 = new kt::nodes::Tracklets(&tracklets_, scan_at_);
  osg::ref_ptr<rt::nodes::OccGrid> og1n = new rt::nodes::OccGrid(og1);
  osg::ref_ptr<rt::nodes::OccGrid> og2n = new rt::nodes::OccGrid(og2, cm);
  osg::ref_ptr<fl::nodes::FlowImage> fin = new fl::nodes::FlowImage(fi);
  osg::ref_ptr<fl::nodes::ClassificationMap> cmn = new fl::nodes::ClassificationMap(cm);
  osg::ref_ptr<osgn::Car> car_node = new osgn::Car(car_path_);

  viewer_->RemoveAllChildren();

  if (view_mode_ == 1) {
    viewer_->AddChild(og1n);
    viewer_->AddChild(tn1);
    viewer_->AddChild(fin);
    viewer_->AddChild(car_node);
  } else if (view_mode_ == 2) {
    viewer_->AddChild(og2n);
    viewer_->AddChild(tn2);
    viewer_->AddChild(fin);
    viewer_->AddChild(car_node);
  } else if (view_mode_ == 3) {
    viewer_->AddChild(pc);
    viewer_->AddChild(cmn);
  }

  if (prev_dm_) {
    viewer_->AddChild(*prev_dm_);
  }
}

void App::ProcessCommands() {
  while (running_) {
    const Command c = command_queue_.Pop(10);

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
