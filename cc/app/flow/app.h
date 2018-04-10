#pragma once

#include <boost/filesystem.hpp>

#include "library/ray_tracing/occ_grid_builder.h"
#include "library/kitti/velodyne_scan.h"
#include "library/kitti/tracklets.h"
#include "library/viewer/viewer.h"

namespace fs = boost::filesystem;
namespace rt = library::ray_tracing;
namespace kt = library::kitti;
namespace vw = library::viewer;

namespace app {
namespace flow {

class App {
 public:
  App(const fs::path &tsf_dir, const std::string &date, int log_num);

  void SetViewer(const std::shared_ptr<vw::Viewer> &viewer);

  void ProccesFrame(int frame_num);

 private:
  static constexpr int kMaxVelodyneScanPoints = 150000;
  static constexpr float kResolution = 0.3;
  static constexpr float kMaxRange = 100.0;

  std::vector<kt::VelodyneScan> scans_;
  kt::Tracklets tracklets_;

  rt::OccGridBuilder og_builder_;

  std::shared_ptr<vw::Viewer> viewer_;

};

} // flow
} // app
