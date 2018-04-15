#include "library/flow/solver.cu.h"

#include "library/timer/timer.h"

namespace library {
namespace flow {

Solver::Solver() {

}

__global__ void FlowKernel(const gu::GpuData<4> dist_sq, gu::GpuData<3> res) {
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

  // Find best distance
  int best_di = 0;
  int best_dj = 0;
  float best_d2 = 99999999.999;

  for (int di=0; di<dist_sq.GetDim(2); di++) {
    for (int dj=0; dj<dist_sq.GetDim(3); dj++) {
      float d2 = dist_sq(i, j, di, dj);

      if (d2 < best_d2) {
        best_di = di;
        best_dj = dj;
        best_d2 = d2;
      }
    }
  }

  res(i, j, 0) = best_di - dist_sq.GetDim(2)/2;
  res(i, j, 1) = best_dj - dist_sq.GetDim(3)/2;

  //best_scores(i, j) = best_d2;
}

void Solver::ComputeFlow(const gu::GpuData<4> &dist_sq, gu::GpuData<3> *res) const {
  library::timer::Timer timer;

  BOOST_ASSERT(res->GetDim(0) == dist_sq.GetDim(0));
  BOOST_ASSERT(res->GetDim(1) == dist_sq.GetDim(1));
  BOOST_ASSERT(res->GetDim(2) == 2);

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
  FlowKernel<<<blocks, threads>>>(dist_sq, *res);
  cudaError_t err = cudaDeviceSynchronize();
  BOOST_ASSERT(err == cudaSuccess);
  printf("Took %5.3f to evaluate flow\n", timer.GetMs());
}

} // namespace tf
} // namespace library
