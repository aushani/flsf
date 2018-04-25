#pragma once

#include <thread>

#include <boost/filesystem.hpp>
#include <boost/optional.hpp>

#include "library/flow/flow_processor.h"
#include "library/flow/nodes/distance_map.h"
#include "library/ray_tracing/occ_grid_builder.h"
#include "library/kitti/velodyne_scan.h"
#include "library/kitti/pose.h"
#include "library/kitti/tracklets.h"
#include "library/viewer/viewer.h"

#include "app/flow/command.h"
#include "app/flow/command_queue.h"

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

  App(const App &app);
  ~App();

  App operator=(const App &app);

  void SetViewer(const std::shared_ptr<vw::Viewer> &viewer);

  void ProcessNext();

  void QueueCommand(const Command &command);

 private:
  std::vector<kt::VelodyneScan> scans_;
  kt::Tracklets tracklets_;
  std::vector<kt::Pose> raw_poses_;
  std::vector<kt::Pose> sm_poses_;

  size_t scan_at_ = 0;

  fl::FlowProcessor flow_processor_;

  fs::path car_path_;

  std::shared_ptr<vw::Viewer> viewer_;
  int view_mode_ = 1;

  boost::optional<osg::ref_ptr<fl::nodes::DistanceMap> > prev_dm_;

  CommandQueue command_queue_;
  std::thread command_thread_;
  bool running_ = true;

  void LoadVelodyneData(const fs::path &tsf_dir, const std::string &date, int log_num);
  void LoadTrackletData(const fs::path &tsf_dir, const std::string &date, int log_num);
  void LoadPoses(const fs::path &tsf_dir, const std::string &date, int log_num);

  void ProcessFrame(int frame_num);
  void UpdateViewer();

  void ProcessCommands();

  void HandleClick(const Command &command);
  void HandleViewMode(const Command &command);
};

} // flow
} // app
