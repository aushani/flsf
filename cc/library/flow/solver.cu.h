#pragma once

#include "library/gpu_util/gpu_data.cu.h"

#include "library/flow/flow_image.h"

namespace gu = library::gpu_util;

namespace library {
namespace flow {

class Solver {
 public:
  Solver();

  FlowImage ComputeFlow(const gu::GpuData<4, float> &dist_sq, gu::GpuData<3, int> *res) const;
};

} // namespace tf
} // namespace library
