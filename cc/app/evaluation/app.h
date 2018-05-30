#pragma once

#include <boost/filesystem.hpp>

#include "library/flow/flow_processor.h"
#include "library/kitti/velodyne_scan.h"
#include "library/kitti/pose.h"
#include "library/kitti/tracklets.h"
#include "library/kitti/camera_cal.h"

namespace fs = boost::filesystem;
namespace kt = library::kitti;
namespace fl = library::flow;

namespace app {
namespace evaluation {

class App {
 public:
  App(const fs::path &tsf_dir, const std::string &date, int log_num);

  void Run(const fs::path &save_path);

  void SetSmoothing(float val);

 private:
  std::vector<kt::VelodyneScan> scans_;
  kt::Tracklets tracklets_;
  std::vector<kt::Pose> raw_poses_;
  std::vector<kt::Pose> sm_poses_;

  kt::CameraCal camera_cal_;

  fl::FlowProcessor flow_processor_;

  void LoadVelodyneData(const fs::path &tsf_dir, const std::string &date, int log_num);
  void LoadTrackletData(const fs::path &tsf_dir, const std::string &date, int log_num);
  void LoadPoses(const fs::path &tsf_dir, const std::string &date, int log_num);
};

} // evaluation
} // app
