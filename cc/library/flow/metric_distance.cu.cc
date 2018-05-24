#include "library/flow/metric_distance.cu.h"

#include "library/timer/timer.h"

namespace library {
namespace flow {

MetricDistance::MetricDistance() {

}

__global__ void DistanceKernel(const gu::GpuData<3, float> d1, const gu::GpuData<3, float> d2, gu::GpuData<4, float> res) {
  const int bidx = blockIdx.x;
  const int bidy = blockIdx.y;

  const int tidx = threadIdx.x;
  const int tidy = threadIdx.y;

  const int search_size_x = res.GetDim(2);
  const int search_size_y = res.GetDim(3);

  // Get the (di, dj) we need to evaluate
  const int off_x = search_size_x / 2;
  const int off_y = search_size_y / 2;
  const int d_i = tidx - off_x;
  const int d_j = tidy - off_y;

  // Location in d1
  const int i1 = bidx;
  const int j1 = bidy;

  // Location in d2
  const int i2 = i1 + d_i;
  const int j2 = j1 + d_j;

  // Find distance
  float dist = -1.0;

  if (d2.InRange(i2, j2, 0)) {
    float dist_sq = 0.0;

    for (int k=0; k<d1.GetDim(2); k++) {
      float dx = d1(i1, j1, k) - d2(i2, j2, k);

      dist_sq += dx*dx;
    }

    dist = sqrt(dist_sq);
  }

  res(i1, j1, tidx, tidy) = dist;
}

void MetricDistance::ComputeDistance(const gu::GpuData<3, float> &d1,
                                     const gu::GpuData<3, float> &d2,
                                     gu::GpuData<4, float> *res) const {
  library::timer::Timer timer;

  BOOST_ASSERT(d1.GetDim(0) == d2.GetDim(0));
  BOOST_ASSERT(d1.GetDim(1) == d2.GetDim(1));
  BOOST_ASSERT(d1.GetDim(2) == d2.GetDim(2));

  BOOST_ASSERT(res->GetDim(0) == d1.GetDim(0));
  BOOST_ASSERT(res->GetDim(1) == d1.GetDim(1));

  dim3 blocks;
  blocks.x = d1.GetDim(0);
  blocks.y = d1.GetDim(1);
  blocks.z = 1;

  dim3 threads;
  threads.x = res->GetDim(2);
  threads.y = res->GetDim(3);
  threads.z = 1;

  timer.Start();
  DistanceKernel<<<blocks, threads>>>(d1, d2, *res);
  cudaError_t err = cudaDeviceSynchronize();
  BOOST_ASSERT(err == cudaSuccess);
  printf("Took %5.3f to evaluate distance\n", timer.GetMs());
}

} // namespace tf
} // namespace library
