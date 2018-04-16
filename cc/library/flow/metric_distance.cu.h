#pragma once

#include "library/gpu_util/gpu_data.cu.h"

namespace gu = library::gpu_util;

namespace library {
namespace flow {

class MetricDistance {
 public:
  MetricDistance();

  void ComputeDistance(const gu::GpuData<3, float> &d1,
                       const gu::GpuData<3, float> &d2,
                       gu::GpuData<4, float> *res) const;
};

} // namespace tf
} // namespace library
