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
  // Load data
  LoadVelodyneData(tsf_dir, date, log_num);
  LoadTrackletData(tsf_dir, date, log_num);
  LoadPoses(tsf_dir, date, log_num);

  // Initialize flow processor with first scan
  flow_processor_.Initialize(scans_[0]);

  printf("Done loading data\n");
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

    rt::OccGrid og = flow_processor_.GetLastOccGrid();
    fl::FlowImage fi = flow_processor_.GetFlowImage();
    fl::ClassificationMap cm = flow_processor_.GetClassificationMap();

    osg::ref_ptr<kt::nodes::PointCloud> pc = new kt::nodes::PointCloud(scan);
    osg::ref_ptr<kt::nodes::Tracklets> tn = new kt::nodes::Tracklets(&tracklets_, frame_num);
    osg::ref_ptr<rt::nodes::OccGrid> ogn = new rt::nodes::OccGrid(og, cm);
    osg::ref_ptr<fl::nodes::FlowImage> fin = new fl::nodes::FlowImage(fi, og.GetResolution());
    osg::ref_ptr<fl::nodes::ClassificationMap> cmn = new fl::nodes::ClassificationMap(cm, og.GetResolution());
    //osg::ref_ptr<osgn::Car> car_node = new osgn::Car(car_path);

    viewer_->RemoveAllChildren();

    //viewer_->AddChild(pc);
    viewer_->AddChild(tn);
    viewer_->AddChild(ogn);
    viewer_->AddChild(fin);
    //viewer_->AddChild(cmn);
    //viewer_->AddChild(car_node);

    printf("Done\n");
  }
}

void App::ProcessNext() {
  scan_at_++;
  ProcessFrame(scan_at_);
}

} // flow
} // app
