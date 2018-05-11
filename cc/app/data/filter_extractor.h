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

class FilterExtractor {
 public:
  FilterExtractor(const fs::path &base_path, const fs::path &save_path);

  void Run();

 private:
  std::vector<kt::VelodyneScan> scans_;
  kt::Tracklets tracklets_;
  kt::CameraCal camera_cal_;

  rt::OccGridBuilder og_builder_;

  std::ofstream save_file_;

  void WriteOccGrid(const rt::OccGrid &og);
  void WriteFilter(int frame);
};

} // data
} // app
