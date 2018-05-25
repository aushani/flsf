#pragma once

#include <map>

#include <boost/filesystem.hpp>

#include "library/flow/flow_image.h"
#include "library/flow/filter_map.h"

#include "library/kitti/camera_cal.h"
#include "library/kitti/object_class.h"
#include "library/kitti/pose.h"
#include "library/kitti/tracklets.h"

#include "library/evaluation/error_stats.h"

namespace kt = library::kitti;
namespace fl = library::flow;
namespace fs = boost::filesystem;

namespace library {
namespace evaluation {

class FlowImageEvaluator {
 public:
  FlowImageEvaluator(const kt::Tracklets &tracklets, const kt::CameraCal &cc, const std::vector<kt::Pose> &poses);

  void Evaluate(const fl::FlowImage &flow_image, const fl::FilterMap &fm, int from, int to);

  void Clear();

  void WriteErrors(const fs::path &path) const;

 private:
  kt::Tracklets tracklets_;
  kt::CameraCal camera_cal_;
  std::vector<kt::Pose> poses_;

  std::map<kt::ObjectClass, ErrorStats> errors_;
};

} // namespace evaluation
} // namespace library
