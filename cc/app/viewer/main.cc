#include <iostream>
#include <osg/ArgumentParser>

#include "library/kitti/velodyne_scan.h"
#include "library/kitti/tracklets.h"
#include "library/kitti/nodes/point_cloud.h"
#include "library/kitti/nodes/tracklets.h"
//#include "library/osg_nodes/car.h"
#include "library/ray_tracing/occ_grid_builder.h"
#include "library/ray_tracing/nodes/occ_grid.h"
#include "library/timer/timer.h"
#include "library/viewer/viewer.h"
//#include "library/gpu_util/util.h"

#include "app/viewer/simple_handler.h"

namespace kt = library::kitti;
namespace rt = library::ray_tracing;
namespace vw = library::viewer;
namespace avw = app::viewer;

kt::VelodyneScan LoadVelodyneScan(const std::string &kitti_log_dir,
                                  const std::string &kitti_log_date,
                                  int log_num,
                                  int frame_num) {
  char fn[1000];
  sprintf(fn, "%s/%s/%s_drive_%04d_sync/velodyne_points/data/%010d.bin",
      kitti_log_dir.c_str(), kitti_log_date.c_str(), kitti_log_date.c_str(), log_num, frame_num);

  return kt::VelodyneScan(fn);
}

kt::Tracklets LoadTracklets(const std::string &kitti_log_dir,
                            const std::string &kitti_log_date,
                            int log_num) {
  // Load Tracklets
  char fn[1000];
  sprintf(fn, "%s/%s/%s_drive_%04d_sync/tracklet_labels.xml",
      kitti_log_dir.c_str(), kitti_log_date.c_str(), kitti_log_date.c_str(), log_num);
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

  au->addCommandLineOption("--kitti-log-dir <dirname>", "KITTI data directory", "~/data/tsf_data/kittidata/");
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
  std::string kitti_log_dir = home_dir + "/data/tsf_data/kittidata/";
  if (!args.read("--kitti-log-dir", kitti_log_dir)) {
    printf("Using default KITTI log dir: %s\n", kitti_log_dir.c_str());
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

  std::string model_dir = "/home/aushani/data/gen_data_50cm_blurred/training/";
  if (!args.read("--models", model_dir)) {
    printf("no model given, using default dir %s\n", model_dir.c_str());
  }

  // Load velodyne scan
  printf("Loading vel\n");
  kt::VelodyneScan scan = LoadVelodyneScan(kitti_log_dir, kitti_log_date, log_num, frame_num);
  printf("Loading tracklets\n");
  kt::Tracklets tracklets = LoadTracklets(kitti_log_dir, kitti_log_date, log_num);

  printf("Have %ld points\n", scan.GetHits().size());

  // Build occ grid
  rt::OccGridBuilder builder(150000, 0.2, 100.0);

  library::timer::Timer t;
  auto og = builder.GenerateOccGrid(scan.GetHits());
  printf("Took %5.3f ms to build occ grid\n", t.GetMs());

  printf("Occ grid has %ld voxels\n", og.GetLocations().size());

  vw::Viewer v(&args);

  osg::ref_ptr<kt::nodes::PointCloud> pc = new kt::nodes::PointCloud(scan);
  osg::ref_ptr<rt::nodes::OccGrid> ogn = new rt::nodes::OccGrid(og);
  osg::ref_ptr<kt::nodes::Tracklets> tn = new kt::nodes::Tracklets(&tracklets, frame_num);
  osg::ref_ptr<avw::SimpleHandler> ph = new avw::SimpleHandler(tracklets, frame_num);

  v.AddChild(pc);
  v.AddChild(ogn);
  //v.AddChild(tn);
  v.AddHandler(ph);

  osg::ref_ptr<osg::MatrixTransform> xform_car = new osg::MatrixTransform();
  osg::Matrixd D(osg::Quat(M_PI, osg::Vec3d(1, 0, 0)));
  D.postMultTranslate(osg::Vec3d(-1, 0, -1.2));
  xform_car->setMatrix(D);
  //xform_car->addChild(new osgn::Car());

  v.AddChild(xform_car);

  v.Start();

  return 0;
}
