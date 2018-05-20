#pragma once

#include <vector>

#include <boost/filesystem.hpp>

#include "library/flow/flow_image.h"

#include "library/kitti/camera_cal.h"
#include "library/kitti/pose.h"
#include "library/kitti/tracklets.h"

namespace kt = library::kitti;
namespace fl = library::flow;
namespace fs = boost::filesystem;

namespace library {
namespace evaluation {

class FlowImageEvaluator {
 public:
  FlowImageEvaluator(const kt::Tracklets &tracklets, const kt::CameraCal &cc, const std::vector<kt::Pose> &poses);

  void Evaluate(const fl::FlowImage &flow_image, int from, int to);

  void Clear();

  void WriteErrors(const fs::path &path);

 private:
  kt::Tracklets tracklets_;
  kt::CameraCal camera_cal_;
  std::vector<kt::Pose> poses_;

  std::vector<float> errors_;

  double total_err_ = 0.0;
  int count_ = 0;
};

} // namespace evaluation
} // namespace library
