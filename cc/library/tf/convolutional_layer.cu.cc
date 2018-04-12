#include "library/tf/convolutional_layer.h"

#include <boost/assert.hpp>

#include "library/timer/timer.h"

namespace library {
namespace tf {

ConvolutionalLayer::ConvolutionalLayer(const gu::GpuData<4> weights, const gu::GpuData<1> biases) :
 weights_(weights),
 biases_(biases) {
  BOOST_ASSERT(weights_.GetDim(3) == biases_.GetDim(0));
  BOOST_ASSERT(weights_.GetDim(3) <= kMaxOutputs);
}

__global__ void ApplyKernel(const gu::GpuData<4> weights, const gu::GpuData<1> biases, const gu::GpuData<3> input, gu::GpuData<3> output, bool relu) {
  // Figure out which hit this thread is processing
  const int bidx = blockIdx.x;
  const int bidy = blockIdx.y;

  const int tidx = threadIdx.x;
  const int tidy = threadIdx.y;

  const int threads_x = blockDim.x;
  const int threads_y = blockDim.y;

  const int idx_i = tidx + bidx * threads_x;
  const int idx_j = tidy + bidy * threads_y;

  if (idx_i < 0 || idx_i >= input.GetDim(0)) {
    return;
  }

  if (idx_j < 0 || idx_j >= input.GetDim(1)) {
    return;
  }

  float res[ConvolutionalLayer::kMaxOutputs];

  // Bias
  for (int k=0; k<biases.GetDim(0); k++) {
    //output(idx_i, idx_j, k) = biases(k);
    res[k] = biases(k);
  }

  // Weights
  for (int i=0; i<weights.GetDim(0); i++) {
    int ii = i + idx_i - weights.GetDim(0)/2;
    if (ii < 0 || ii >= input.GetDim(0)) {
      continue;
    }

    for (int j=0; j<weights.GetDim(1); j++) {
      int jj = j + idx_j - weights.GetDim(1)/2;
      if (jj < 0 || jj >= input.GetDim(1)) {
        continue;
      }

      for (int k=0; k<weights.GetDim(2); k++) {
        float val = input(ii, jj, k);

        for (int layer=0; layer<weights.GetDim(3); layer++) {
          float w = weights(i, j, k, layer);

          res[layer] += val * w;
          //output(idx_i, idx_j, k) += val * w;
        }
      }
    }
  }

  // RELU Activation
  if (relu) {
    for (int k=0; k<output.GetDim(2); k++) {
      //if (output(idx_i, idx_j, k) < 0) {
      //  output(idx_i, idx_j, k) = 0.0;
      //}
      if (res[k] < 0) {
        res[k] = 0.0;
      }
    }
  }

  // Write result
  for (int k=0; k<output.GetDim(2); k++) {
    output(idx_i, idx_j, k) = res[k];
  }
}

void ConvolutionalLayer::Apply(const gu::GpuData<3> &input, gu::GpuData<3> *output) {
  library::timer::Timer t;

  // Check dimensions
  BOOST_ASSERT(input.GetDim(0) == output->GetDim(0));
  BOOST_ASSERT(input.GetDim(1) == output->GetDim(1));

  BOOST_ASSERT(input.GetDim(2) == weights_.GetDim(2));
  BOOST_ASSERT(output->GetDim(2) == weights_.GetDim(3));

  dim3 threads;
  threads.x = 32;
  threads.y = 1;
  threads.z = 1;

  dim3 blocks;
  blocks.x = std::ceil(static_cast<float>(input.GetDim(0))/threads.x);
  blocks.y = std::ceil(static_cast<float>(input.GetDim(1))/threads.y);
  blocks.z = 1;

  printf("\tRunning kernel with %dx%d threads and %dx%d blocks\n",
        threads.x, threads.y, blocks.x, blocks.y);

  t.Start();
  ApplyKernel<<<blocks, threads>>>(weights_, biases_, input, *output, true);
  cudaError_t err = cudaDeviceSynchronize();
  BOOST_ASSERT(err == cudaSuccess);
  printf("\tKernel took %5.3f ms\n", t.GetMs());
}

} // namespace tf
} // namespace library
