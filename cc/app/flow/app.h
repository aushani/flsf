#pragma once

#include <thread>
#include <memory>

#include <boost/filesystem.hpp>

#include "library/evaluation/flow_image_evaluator.h"
#include "library/flow/flow_processor.h"
#include "library/ray_tracing/occ_grid_builder.h"
#include "library/kitti/velodyne_scan.h"
#include "library/kitti/pose.h"
#include "library/kitti/tracklets.h"
#include "library/kitti/camera_cal.h"
#include "library/viewer/viewer.h"

#include "app/flow/command.h"
#include "app/flow/command_queue.h"
#include "app/flow/node_manager.h"

namespace fs = boost::filesystem;
namespace rt = library::ray_tracing;
namespace kt = library::kitti;
namespace vw = library::viewer;
namespace fl = library::flow;
namespace ev = library::evaluation;

namespace app {
namespace flow {

class App {
 public:
  App(const fs::path &tsf_dir, const std::string &date, const fs::path &network_path, int log_num, int frame_num=0);

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

  kt::CameraCal camera_cal_;

  size_t scan_at_ = 0;

  fl::FlowProcessor flow_processor_;

  NodeManager node_manager_;

  CommandQueue command_queue_;
  std::thread command_thread_;
  bool running_ = true;

  std::unique_ptr<ev::FlowImageEvaluator> fi_eval_;

  void LoadVelodyneData(const fs::path &tsf_dir, const std::string &date, int log_num);
  void LoadTrackletData(const fs::path &tsf_dir, const std::string &date, int log_num);
  void LoadPoses(const fs::path &tsf_dir, const std::string &date, int log_num);

  void Process();
  void Refresh();

  void ProcessCommands();

  void HandleClick(const Command &command);
  void HandleViewMode(const Command &command);
  void HandleClearDistanceMap(const Command &command);
  void HandleIterations(const Command &command);
  void HandleRefresh(const Command &command);
};

} // flow
} // app
