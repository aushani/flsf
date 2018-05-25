#pragma once

#include "library/gpu_util/gpu_data.cu.h"

#include "library/flow/flow_image.h"

namespace gu = library::gpu_util;

namespace library {
namespace flow {

class Solver {
 public:
  Solver(int nx, int ny, int n_window);

  FlowImage ComputeFlow(const gu::GpuData<4, float> &dist_sq,
                        const gu::GpuData<2, float> &p_background,
                        const gu::GpuData<2, int> &occ_mask,
                        float resolution,
                        int iters);

 private:
  static constexpr float kSmoothing_ = 0.01;

  const int           nx_;
  const int           ny_;

  const int n_window_;

  gu::GpuData<2, float>  energy_;
  gu::GpuData<2, float>  energy_hat_;
  gu::GpuData<2, int>    energy_hat_valid_;
  gu::GpuData<3, int>    flow_est_;
  gu::GpuData<2, int>    flow_valid_;
};

} // namespace tf
} // namespace library
