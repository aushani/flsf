#pragma once

#include <boost/optional.hpp>
#include <cudnn.h>

#include "library/gpu_util/gpu_data.cu.h"

#include "library/tf/activation.cu.h"

namespace gu = library::gpu_util;

namespace library {
namespace tf {

class ConvolutionalLayer {
 public:
  ConvolutionalLayer(size_t input_height, size_t input_width, const gu::GpuData<4, float> &weights, const gu::GpuData<1, float> &biases, bool zero_padding);

  void SetActivation(const Activation &op);

  int GetInputHeight() const;
  int GetInputWidth() const;
  int GetOutputHeight() const;
  int GetOutputWidth() const;
  int GetOutputLayers() const;

  void Apply(const gu::GpuData<3, float> &input, gu::GpuData<3, float> *output);

 private:
  const int input_height_;
  const int input_width_;

  const int kernel_height_;
  const int kernel_width_;

  const int output_height_;
  const int output_width_;

  const int input_channels_;
  const int output_channels_;

  gu::GpuData<4, float> weights_;
  gu::GpuData<1, float> biases_;

  cudnnTensorDescriptor_t input_descriptor_;
  cudnnTensorDescriptor_t output_descriptor_;

  cudnnTensorDescriptor_t biases_descriptor_;

  cudnnFilterDescriptor_t kernel_descriptor_;

  cudnnConvolutionDescriptor_t convolution_desriptor_;
  cudnnConvolutionFwdAlgo_t convolution_algorithm_;

  Activation activation_ = Activation::NONE;

  std::shared_ptr<gu::GpuData<1, uint8_t> > d_workspace_;
};

} // namespace tf
} // namespace library
