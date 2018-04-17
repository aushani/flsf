#pragma once

#include <boost/filesystem.hpp>

#include "library/flow/flow_processor.h"
#include "library/ray_tracing/occ_grid_builder.h"
#include "library/kitti/velodyne_scan.h"
#include "library/kitti/pose.h"
#include "library/kitti/tracklets.h"
#include "library/viewer/viewer.h"

namespace fs = boost::filesystem;
namespace rt = library::ray_tracing;
namespace kt = library::kitti;
namespace vw = library::viewer;
namespace fl = library::flow;

namespace app {
namespace flow {

class App {
 public:
  App(const fs::path &tsf_dir, const std::string &date, int log_num);

  void SetViewer(const std::shared_ptr<vw::Viewer> &viewer);

  void ProcessNext();

 private:
  std::vector<kt::VelodyneScan> scans_;
  kt::Tracklets tracklets_;
  std::vector<kt::Pose> raw_poses_;
  std::vector<kt::Pose> sm_poses_;

  size_t scan_at_ = 0;

  fl::FlowProcessor flow_processor_;

  std::shared_ptr<vw::Viewer> viewer_;

  void LoadVelodyneData(const fs::path &tsf_dir, const std::string &date, int log_num);
  void LoadTrackletData(const fs::path &tsf_dir, const std::string &date, int log_num);
  void LoadPoses(const fs::path &tsf_dir, const std::string &date, int log_num);

  void ProcessFrame(int frame_num);
};

} // flow
} // app
