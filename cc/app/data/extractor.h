#pragma once

#include <iostream>
#include <fstream>
#include <random>

#include <boost/filesystem.hpp>

#include "library/kitti/camera_cal.h"
#include "library/kitti/pose.h"
#include "library/kitti/tracklets.h"
#include "library/kitti/velodyne_scan.h"
#include "library/params/params.h"
#include "library/ray_tracing/occ_grid.h"
#include "library/ray_tracing/occ_grid_builder.h"


namespace fs = boost::filesystem;
namespace kt = library::kitti;
namespace ps = library::params;
namespace rt = library::ray_tracing;

namespace app {
namespace data {

class Extractor {
 public:
  Extractor(const fs::path &base_path, const fs::path &save_path);

  void Run();

 private:
  std::vector<kt::VelodyneScan> scans_;
  kt::Tracklets tracklets_;
  std::vector<kt::Pose> sm_poses_;
  kt::CameraCal camera_cal_;

  rt::OccGridBuilder og_builder_;

  std::random_device random_device_;
  std::mt19937 random_generator_;

  std::ofstream save_file_;

  void ProcessOccGrids(const rt::OccGrid &og1, const rt::OccGrid &og2, int idx1, int idx2);
  void WriteOccGrid(const rt::OccGrid &og);
  void WriteFilter(int frame);
  void WriteFlow(int frame1, int frame2);
};

} // data
} // app
