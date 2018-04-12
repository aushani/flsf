#include "app/flow/app.h"

#include <boost/format.hpp>

#include "library/kitti/nodes/point_cloud.h"
#include "library/kitti/nodes/tracklets.h"
#include "library/osg_nodes/car.h"
#include "library/ray_tracing/nodes/occ_grid.h"
#include "library/timer/timer.h"

namespace osgn = library::osg_nodes;

namespace app {
namespace flow {

App::App(const fs::path &tsf_dir, const std::string &date, int log_num) :
 og_builder_(kMaxVelodyneScanPoints, kResolution, kMaxRange),
 network_(tf::Network::LoadNetwork("/home/aushani/koopa_training/")) {

  // Load data
  std::string dir_name = (boost::format("%s_drive_%04d_sync") % date % log_num).str();

  printf("Loading velodyne data...\n");
  int frame_num = 0;
  while (true) {
    std::string fn = (boost::format("%010d.bin") % frame_num).str();
    fs::path path = tsf_dir / "kittidata" / date / dir_name / "velodyne_points" / "data" / fn;

    if (!fs::exists(path)) {
      break;
    }

    //printf("Load %s\n", path.string().c_str());
    scans_.emplace_back(path.string());

    frame_num++;
  }

  printf("Loading tracklet data...\n");
  fs::path tracklet_file = tsf_dir / "kittidata" / date / dir_name / "tracklet_labels.xml";
  tracklets_.loadFromFile(tracklet_file.string());

  printf("Done loading data\n");
}

void App::SetViewer(const std::shared_ptr<vw::Viewer> &viewer) {
  viewer_ = viewer;
}

void App::ProcessFrame(int frame_num) {
  library::timer::Timer timer;

  const auto &scan = scans_[frame_num];

  timer.Start();
  auto og = og_builder_.GenerateOccGrid(scan.GetHits());
  printf("Took %5.3f ms to get occ grid\n", timer.GetMs());

  timer.Start();
  network_->SetInput(og);
  printf("Took %5.3f ms to send occ grid to network\n", timer.GetMs());

  timer.Start();
  network_->Apply();
  printf("Took %5.3f ms to run network\n", timer.GetMs());

  if (viewer_) {
    printf("Update viewer\n");

    osg::ref_ptr<kt::nodes::PointCloud> pc = new kt::nodes::PointCloud(scan);
    osg::ref_ptr<rt::nodes::OccGrid> ogn = new rt::nodes::OccGrid(og);
    osg::ref_ptr<kt::nodes::Tracklets> tn = new kt::nodes::Tracklets(&tracklets_, frame_num);
    //osg::ref_ptr<osgn::Car> car_node = new osgn::Car(car_path);

    viewer_->RemoveAllChildren();

    viewer_->AddChild(pc);
    viewer_->AddChild(ogn);
    viewer_->AddChild(tn);
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
