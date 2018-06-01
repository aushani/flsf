#include "library/tf/convolutional_layer.cu.h"

#include <boost/assert.hpp>
#include <thrust/transform.h>

#include "library/timer/timer.h"

#include "library/tf/util.cu.h"

namespace library {
namespace tf {

ConvolutionalLayer::ConvolutionalLayer(size_t input_height, size_t input_width, const gu::GpuData<4, float> &weights,
    const gu::GpuData<1, float> &biases, bool zero_padding) :
 input_height_(input_height),
 input_width_(input_width),
 kernel_height_(weights.GetDim(1)),
 kernel_width_(weights.GetDim(2)),
 output_height_(input_height + ((zero_padding) ? 0 : (-kernel_height_ + 1))),
 output_width_(input_width   + ((zero_padding) ? 0 : (-kernel_width_  + 1))),
 input_channels_(weights.GetDim(3)),
 output_channels_(weights.GetDim(0)),
 weights_(weights),
 biases_(biases) {
  BOOST_ASSERT(biases_.GetDim(0) == output_channels_);

  cudnnHandle_t cudnn = GetCudnnHandle();
  cudnnStatus_t status;

  // Input
  status = cudnnCreateTensorDescriptor(&input_descriptor_);
  BOOST_ASSERT(status == CUDNN_STATUS_SUCCESS);

  status = cudnnSetTensor4dDescriptor(input_descriptor_,
                                      CUDNN_TENSOR_NHWC,    // format
                                      CUDNN_DATA_FLOAT,     // data type
                                      1,                    // batch size
                                      input_channels_,      // channels,
                                      input_height_,        // image height
                                      input_width_);        // image width
  BOOST_ASSERT(status == CUDNN_STATUS_SUCCESS);

  status = cudnnCreateTensorDescriptor(&output_descriptor_);
  BOOST_ASSERT(status == CUDNN_STATUS_SUCCESS);

  // Output
  status = cudnnSetTensor4dDescriptor(output_descriptor_,
                                      CUDNN_TENSOR_NHWC,    // format
                                      CUDNN_DATA_FLOAT,     // data type
                                      1,                    // batch size
                                      output_channels_,     // channels,
                                      output_height_,       // image height
                                      output_width_);       // image width
  BOOST_ASSERT(status == CUDNN_STATUS_SUCCESS);

  // Bias
  status = cudnnCreateTensorDescriptor(&biases_descriptor_);
  BOOST_ASSERT(status == CUDNN_STATUS_SUCCESS);

  status = cudnnSetTensor4dDescriptor(biases_descriptor_,
                                      CUDNN_TENSOR_NHWC,    // format
                                      CUDNN_DATA_FLOAT,     // data type
                                      1,                    // batch size
                                      output_channels_,     // channels
                                      1,                    // bias height
                                      1);                   // bias width
  BOOST_ASSERT(status == CUDNN_STATUS_SUCCESS);

  // Kernel
  status = cudnnCreateFilterDescriptor(&kernel_descriptor_);
  BOOST_ASSERT(status == CUDNN_STATUS_SUCCESS);

  status = cudnnSetFilter4dDescriptor(kernel_descriptor_,
                                      CUDNN_DATA_FLOAT,     // data type
                                      CUDNN_TENSOR_NHWC,    // format
                                      output_channels_,     // out channels
                                      input_channels_,      // in channels
                                      kernel_height_,       // kernel height
                                      kernel_width_);       // kernel width
  BOOST_ASSERT(status == CUDNN_STATUS_SUCCESS);

  status = cudnnCreateConvolutionDescriptor(&convolution_desriptor_);
  BOOST_ASSERT(status == CUDNN_STATUS_SUCCESS);

  // Convolution
  status = cudnnSetConvolution2dDescriptor(convolution_desriptor_,
                                           zero_padding ? kernel_height_/2 : 0,           // pad height
                                           zero_padding ? kernel_width_/2  : 0,            // pad weight
                                           1,                          // vertical stride
                                           1,                          // horizontal stride
                                           1,                          // dialation height
                                           1,                          // dialation width
                                           CUDNN_CROSS_CORRELATION,    // mode
                                           CUDNN_DATA_FLOAT);          // compute type
  BOOST_ASSERT(status == CUDNN_STATUS_SUCCESS);

  // Convolution algorithm
  status = cudnnGetConvolutionForwardAlgorithm(cudnn,
                                               input_descriptor_,
                                               kernel_descriptor_,
                                               convolution_desriptor_,
                                               output_descriptor_,
                                               CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                               0,                                       // memory limit in bytes
                                               &convolution_algorithm_);
  BOOST_ASSERT(status == CUDNN_STATUS_SUCCESS);

  size_t workspace_bytes = 0;
  status = cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                                   input_descriptor_,
                                                   kernel_descriptor_,
                                                   convolution_desriptor_,
                                                   output_descriptor_,
                                                   convolution_algorithm_,
                                                   &workspace_bytes);
  BOOST_ASSERT(status == CUDNN_STATUS_SUCCESS);

  if (workspace_bytes <= 0) {
    workspace_bytes = 1;
  }

  d_workspace_ = std::make_shared<gu::GpuData<1, uint8_t> >(static_cast<int>(workspace_bytes));
}

void ConvolutionalLayer::Apply(const gu::GpuData<3, float> &input, gu::GpuData<3, float> *output) {
  library::timer::Timer t;

  // Check dimensions
  BOOST_ASSERT(input.GetDim(0) == input_height_);
  BOOST_ASSERT(input.GetDim(1) == input_width_);

  BOOST_ASSERT(output->GetDim(0) == output_height_);
  BOOST_ASSERT(output->GetDim(1) == output_width_);

  BOOST_ASSERT(input.GetDim(2)   == input_channels_);
  BOOST_ASSERT(output->GetDim(2) == output_channels_);

  const float alpha = 1.0;
  const float beta = 0.0;
  const float beta2 = 1.0;
  //const float alpha2 = 0.0;

  cudnnHandle_t cudnn = GetCudnnHandle();
  cudnnStatus_t status;
  cudaError_t err;

  t.Start();
  status = cudnnConvolutionForward(cudnn,
                                   &alpha,
                                   input_descriptor_,
                                   input.GetRawPointer(),
                                   kernel_descriptor_,
                                   weights_.GetRawPointer(),
                                   convolution_desriptor_,
                                   convolution_algorithm_,
                                   d_workspace_->GetRawPointer(),
                                   d_workspace_->Size(),
                                   &beta,
                                   output_descriptor_,
                                   output->GetRawPointer());
  BOOST_ASSERT(status == CUDNN_STATUS_SUCCESS);
  err = cudaDeviceSynchronize();
  BOOST_ASSERT(err == cudaSuccess);
  printf("\tConvolution took %5.3f ms\n", t.GetMs());

  t.Start();
  status = cudnnAddTensor(cudnn,
                          &alpha,
                          biases_descriptor_,
                          biases_.GetRawPointer(),
                          &beta2,
                          output_descriptor_,
                          output->GetRawPointer());
  BOOST_ASSERT(status == CUDNN_STATUS_SUCCESS);
  err = cudaDeviceSynchronize();
  BOOST_ASSERT(err == cudaSuccess);
  printf("\tBiases took %5.3f ms\n", t.GetMs());

  if (activation_ == Activation::LEAKY_RELU) {
    t.Start();
    thrust::transform(output->Begin(), output->End(), output->Begin(), leaky_relu());
    err = cudaDeviceSynchronize();
    BOOST_ASSERT(err == cudaSuccess);
    printf("\tLeaky RELU activation took %5.3f ms\n", t.GetMs());
  }
}

void ConvolutionalLayer::SetActivation(const Activation &op) {
  activation_ = op;
}

int ConvolutionalLayer::GetInputLayers() const {
  return input_channels_;
}

int ConvolutionalLayer::GetOutputLayers() const {
  return output_channels_;
}

int ConvolutionalLayer::GetInputHeight() const {
  return input_height_;
}

int ConvolutionalLayer::GetInputWidth() const {
  return input_width_;
}

int ConvolutionalLayer::GetOutputHeight() const {
  return output_height_;
}

int ConvolutionalLayer::GetOutputWidth() const {
  return output_width_;
}

} // namespace tf
} // namespace library
