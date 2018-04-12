#pragma once

#include "library/gpu_util/gpu_data.cu.h"

namespace gu = library::gpu_util;

namespace library {
namespace tf {

class ConvolutionalLayer {
 public:
  ConvolutionalLayer(const gu::GpuData<4> weights, const gu::GpuData<1> biases);

  void Apply(const gu::GpuData<3> &input, gu::GpuData<3> *output);

  static constexpr int kMaxOutputs = 200;

 private:
  gu::GpuData<4> weights_;
  gu::GpuData<1> biases_;
};

} // namespace tf
} // namespace library
