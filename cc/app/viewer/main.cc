#include <iostream>
#include <osg/ArgumentParser>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>

#include "library/kitti/velodyne_scan.h"
#include "library/kitti/tracklets.h"
#include "library/kitti/nodes/point_cloud.h"
#include "library/kitti/nodes/tracklets.h"
#include "library/params/params.h"
#include "library/osg_nodes/car.h"
#include "library/ray_tracing/occ_grid_builder.h"
#include "library/ray_tracing/nodes/occ_grid.h"
#include "library/timer/timer.h"
#include "library/viewer/viewer.h"

#include "app/viewer/simple_handler.h"

namespace kt = library::kitti;
namespace ps = library::params;
namespace rt = library::ray_tracing;
namespace vw = library::viewer;
namespace fs = boost::filesystem;
namespace osgn = library::osg_nodes;
namespace avw = app::viewer;

kt::VelodyneScan LoadVelodyneScan(const std::string &tsf_data_dir,
                                  const std::string &kitti_log_date,
                                  int log_num,
                                  int frame_num) {
  char fn[1000];
  sprintf(fn, "%s/kittidata/%s/%s_drive_%04d_sync/velodyne_points/data/%010d.bin",
      tsf_data_dir.c_str(), kitti_log_date.c_str(), kitti_log_date.c_str(), log_num, frame_num);

  return kt::VelodyneScan(fn);
}

kt::Tracklets LoadTracklets(const std::string &tsf_data_dir,
                            const std::string &kitti_log_date,
                            int log_num) {
  // Load Tracklets
  char fn[1000];
  sprintf(fn, "%s/kittidata/%s/%s_drive_%04d_sync/tracklet_labels.xml",
      tsf_data_dir.c_str(), kitti_log_date.c_str(), kitti_log_date.c_str(), log_num);
  kt::Tracklets tracklets;
  if (!tracklets.loadFromFile(fn)) {
    printf("Could not load tracklets from %s\n", fn);
  }

  return tracklets;
}

int main(int argc, char** argv) {
  osg::ArgumentParser args(&argc, argv);
  osg::ApplicationUsage* au = args.getApplicationUsage();

  // report any errors if they have occurred when parsing the program arguments.
  if (args.errors()) {
    args.writeErrorMessages(std::cout);
    return EXIT_FAILURE;
  }

  au->setApplicationName(args.getApplicationName());
  au->setCommandLineUsage( args.getApplicationName() + " [options]");
  au->setDescription(args.getApplicationName() + " viewer");

  au->addCommandLineOption("--tsf-data-dir <dirname>", "TSF data directory", "~/data/tsf_data/");
  au->addCommandLineOption("--kitti-log-date <dirname>", "KITTI date", "2011_09_26");
  au->addCommandLineOption("--log-num <num>", "KITTI log number", "18");
  au->addCommandLineOption("--frame-num <num>", "KITTI frame number", "0");
  //au->addCommandLineOption("--alt", "Run on alt device", "");

  // handle help text
  // call AFTER init viewer so key bindings have been set
  unsigned int helpType = 0;
  if ((helpType = args.readHelpType())) {
    au->write(std::cout, helpType);
    return EXIT_SUCCESS;
  }

  //if (args.read("--alt")) {
  //    library::gpu_util::SetDevice(1);
  //}

  // Read params
  std::string home_dir = getenv("HOME");
  std::string tsf_data_dir = home_dir + "/data/tsf_data";
  if (!args.read("--tsf-data-dir", tsf_data_dir)) {
    printf("Using default tsf data dir: %s\n", tsf_data_dir.c_str());
  }

  std::string kitti_log_date = "2011_09_26";
  if (!args.read("--kitti-log-date", kitti_log_date)) {
    printf("Using default KITTI date: %s\n", kitti_log_date.c_str());
  }

  int log_num = 18;
  if (!args.read("--log-num", log_num)) {
    printf("Using default KITTI log number: %d\n", log_num);
  }

  int frame_num = 0;
  if (!args.read("--frame-num", frame_num)) {
    printf("Using default KITTI frame number: %d\n", frame_num);
  }

  fs::path car_path = fs::path(tsf_data_dir) / "osg_models/lexus/lexus_hs.obj";

  std::string dir_name = (boost::format("%s_drive_%04d_sync") % kitti_log_date % log_num).str();
  fs::path base_path = fs::path(tsf_data_dir) / "kittidata" / kitti_log_date / dir_name;

  // Load velodyne scan
  printf("Loading vel\n");
  kt::VelodyneScan scan = LoadVelodyneScan(tsf_data_dir, kitti_log_date, log_num, frame_num);
  printf("Loading tracklets\n");
  kt::Tracklets tracklets = LoadTracklets(tsf_data_dir, kitti_log_date, log_num);

  printf("Have %ld points\n", scan.GetHits().size());

  // Build occ grid
  rt::OccGridBuilder builder(ps::kMaxVelodyneScanPoints, ps::kResolution, ps::kMaxRange);

  library::timer::Timer t;
  auto og = builder.GenerateOccGrid(scan.GetHits());
  printf("Took %5.3f ms to build occ grid\n", t.GetMs());

  printf("Occ grid has %ld voxels\n", og.GetLocations().size());

  // Start viewer

  vw::Viewer v(&args);

  osg::ref_ptr<kt::nodes::PointCloud> pc = new kt::nodes::PointCloud(scan);
  osg::ref_ptr<rt::nodes::OccGrid> ogn = new rt::nodes::OccGrid();
  osg::ref_ptr<kt::nodes::Tracklets> tn = new kt::nodes::Tracklets(&tracklets, frame_num);
  osg::ref_ptr<avw::SimpleHandler> ph = new avw::SimpleHandler(base_path);
  osg::ref_ptr<osgn::Car> car_node = new osgn::Car(car_path);

  ogn->SetCameraCal(kt::CameraCal(base_path.parent_path()));
  ogn->Update(og);

  ph->SetFrame(frame_num);

  v.AddChild(pc);
  v.AddChild(ogn);
  v.AddChild(tn);
  v.AddChild(car_node);

  v.AddHandler(ph);

  v.Start();

  return 0;
}
