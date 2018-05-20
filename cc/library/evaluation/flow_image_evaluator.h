#pragma once

#include <vector>

#include "library/flow/flow_image.h"

#include "library/kitti/camera_cal.h"
#include "library/kitti/pose.h"
#include "library/kitti/tracklets.h"

namespace kt = library::kitti;
namespace fl = library::flow;

namespace library {
namespace evaluation {

class FlowImageEvaluator {
 public:
  FlowImageEvaluator(const kt::Tracklets &tracklets, const kt::CameraCal &cc, const std::vector<kt::Pose> &poses);

  void Evaluate(const fl::FlowImage &flow_image, int from, int to);

  void Clear();

 private:
  kt::Tracklets tracklets_;
  kt::CameraCal camera_cal_;
  std::vector<kt::Pose> poses_;

  double total_err_ = 0.0;
  int count_ = 0;
};

} // namespace evaluation
} // namespace library
