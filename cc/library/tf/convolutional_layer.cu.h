#pragma once

#include "library/gpu_util/gpu_data.cu.h"

namespace gu = library::gpu_util;

namespace library {
namespace tf {

class ConvolutionalLayer {
 public:
  ConvolutionalLayer(const gu::GpuData<4, float> &weights, const gu::GpuData<1, float> &biases);

  void Apply(const gu::GpuData<3, float> &input, gu::GpuData<3, float> *output);

  static constexpr int kMaxOutputs = 200;

 private:
  gu::GpuData<4, float> weights_;
  gu::GpuData<1, float> biases_;
};

} // namespace tf
} // namespace library
