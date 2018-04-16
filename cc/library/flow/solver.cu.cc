#include "library/flow/solver.cu.h"

#include "library/gpu_util/host_data.cu.h"
#include "library/timer/timer.h"

namespace library {
namespace flow {

Solver::Solver() {
}

__global__ void FlowKernel(const gu::GpuData<4, float> dist_sq,
                           const gu::GpuData<3, float> classification,
                           gu::GpuData<3, int> res) {
  const int bidx = blockIdx.x;
  const int bidy = blockIdx.y;

  const int tidx = threadIdx.x;
  const int tidy = threadIdx.y;

  const int i0 = bidx * blockDim.x;
  const int j0 = bidy * blockDim.y;

  const int i = i0 + tidx;
  const int j = j0 + tidy;

  if (i >= dist_sq.GetDim(0) || j >= dist_sq.GetDim(1)) {
    return;
  }

  // Find p_background
  float denom = 0.0;
  for (int k=0; k<classification.GetDim(2); k++) {
    denom += exp(classification(i, j, k));
  }
  float p_background = exp(classification(i, j, 3))/denom;

  if (p_background > 0.5) {
    res(i, j, 0) = 0;
    res(i, j, 1) = 0;

    return;
  }


  // Find best distance
  int best_di = 0;
  int best_dj = 0;
  float best_d2 = 0;
  bool first = true;

  for (int di=0; di<dist_sq.GetDim(2); di++) {
    for (int dj=0; dj<dist_sq.GetDim(3); dj++) {
      float d2 = dist_sq(i, j, di, dj);

      if (d2 < 0) {
        continue;
      }

      if (d2 < best_d2 || first) {
        best_di = di;
        best_dj = dj;
        best_d2 = d2;

        first = false;
      }
    }
  }

  res(i, j, 0) = best_di - dist_sq.GetDim(2)/2;
  res(i, j, 1) = best_dj - dist_sq.GetDim(3)/2;
}

FlowImage Solver::ComputeFlow(const gu::GpuData<4, float> &dist_sq,
                              const gu::GpuData<3, float> &classification,
                              gu::GpuData<3, int> *res) const {
  library::timer::Timer timer;

  BOOST_ASSERT(res->GetDim(0) == dist_sq.GetDim(0));
  BOOST_ASSERT(res->GetDim(1) == dist_sq.GetDim(1));
  BOOST_ASSERT(res->GetDim(2) == 2);

  BOOST_ASSERT(classification.GetDim(0) == dist_sq.GetDim(0));
  BOOST_ASSERT(classification.GetDim(1) == dist_sq.GetDim(1));

  // Setup thread and block dims
  dim3 threads;
  threads.x = 16;
  threads.y = 16;
  threads.z = 1;

  dim3 blocks;
  blocks.x = std::ceil( static_cast<float>(dist_sq.GetDim(0)) / threads.x);
  blocks.y = std::ceil( static_cast<float>(dist_sq.GetDim(1)) / threads.y);
  blocks.z = 1;

  timer.Start();
  FlowKernel<<<blocks, threads>>>(dist_sq, classification, *res);
  cudaError_t err = cudaDeviceSynchronize();
  BOOST_ASSERT(err == cudaSuccess);
  printf("Took %5.3f to evaluate flow\n", timer.GetMs());

  // Copy from device
  gu::HostData<3, int> h_res = *res;
  FlowImage fi(h_res.GetDim(0), h_res.GetDim(1));

  for (int i=0; i<h_res.GetDim(0); i++) {
    for (int j=0; j<h_res.GetDim(1); j++) {
      int ii = fi.MinX() + i;
      int jj = fi.MinY() + j;

      int flow_x = h_res(i, j, 0);
      int flow_y = h_res(i, j, 1);

      fi.SetFlow(ii, jj, flow_x, flow_y);
    }
  }

  return fi;
}

} // namespace tf
} // namespace library
