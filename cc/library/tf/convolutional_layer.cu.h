#pragma once

#include <cudnn.h>

#include "library/gpu_util/gpu_data.cu.h"

namespace gu = library::gpu_util;

namespace library {
namespace tf {

class ConvolutionalLayer {
 public:
  ConvolutionalLayer(int height, int width, const gu::GpuData<4, float> &weights, const gu::GpuData<1, float> &biases);

  int GetOutputLayers() const;

  void Apply(const gu::GpuData<3, float> &input, gu::GpuData<3, float> *output);

  static constexpr int kMaxOutputs = 200;

 private:
  const int height_;
  const int width_;

  const int input_channels_;
  const int output_channels_;

  const int kernel_height_;
  const int kernel_width_;

  gu::GpuData<4, float> weights_;
  gu::GpuData<1, float> biases_;

  cudnnTensorDescriptor_t input_descriptor_;
  cudnnTensorDescriptor_t output_descriptor_;

  cudnnTensorDescriptor_t biases_descriptor_;

  cudnnFilterDescriptor_t kernel_descriptor_;

  cudnnConvolutionDescriptor_t convolution_desriptor_;
  cudnnConvolutionFwdAlgo_t convolution_algorithm_;

  std::shared_ptr<gu::GpuData<1, uint8_t> > d_workspace_;
};

} // namespace tf
} // namespace library
