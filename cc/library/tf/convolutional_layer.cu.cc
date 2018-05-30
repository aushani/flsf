#include "library/tf/convolutional_layer.cu.h"

#include <boost/assert.hpp>
#include <thrust/transform.h>

#include "library/timer/timer.h"

#include "library/tf/util.cu.h"

namespace library {
namespace tf {

ConvolutionalLayer::ConvolutionalLayer(size_t height, size_t width, const gu::GpuData<4, float> &weights, const gu::GpuData<1, float> &biases) :
 height_(height),
 width_(width),
 input_channels_(weights.GetDim(2)),
 output_channels_(weights.GetDim(3)),
 kernel_height_(weights.GetDim(0)),
 kernel_width_(weights.GetDim(1)),
 weights_(weights),
 biases_(biases) {
  BOOST_ASSERT(weights_.GetDim(3) == biases_.GetDim(0));

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
                                      height_,              // image height
                                      width_);              // image width
  BOOST_ASSERT(status == CUDNN_STATUS_SUCCESS);

  status = cudnnCreateTensorDescriptor(&output_descriptor_);
  BOOST_ASSERT(status == CUDNN_STATUS_SUCCESS);

  // Output
  status = cudnnSetTensor4dDescriptor(output_descriptor_,
                                      CUDNN_TENSOR_NHWC,    // format
                                      CUDNN_DATA_FLOAT,     // data type
                                      1,                    // batch size
                                      output_channels_,     // channels,
                                      height_,              // image height
                                      width_);              // image width
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
                                           kernel_height_/2,           // pad height
                                           kernel_width_/2,            // pad weight
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

struct leaky_relu_op : public thrust::unary_function<float, float> {
  __host__ __device__ float operator()(float x) {
    if (x < 0) {
      return x * 0.2;
    }
    return x;
  }
};

void ConvolutionalLayer::Apply(const gu::GpuData<3, float> &input, gu::GpuData<3, float> *output) {
  library::timer::Timer t;

  // Check dimensions
  BOOST_ASSERT(input.GetDim(0) == output->GetDim(0));
  BOOST_ASSERT(input.GetDim(1) == output->GetDim(1));

  BOOST_ASSERT(input.GetDim(2) == weights_.GetDim(2));
  BOOST_ASSERT(output->GetDim(2) == weights_.GetDim(3));

  BOOST_ASSERT(input.GetDim(0) == height_);
  BOOST_ASSERT(input.GetDim(1) == width_);

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

  t.Start();
  thrust::transform(output->Begin(),
                    output->Begin() + output->Size(),
                    output->Begin(),
                    leaky_relu_op());
  cudaDeviceSynchronize();
  err = cudaDeviceSynchronize();
  BOOST_ASSERT(err == cudaSuccess);
  printf("\tLeaky RELU activation took %5.3f ms\n", t.GetMs());
}

int ConvolutionalLayer::GetOutputLayers() const {
  return biases_.GetDim(0);
}

} // namespace tf
} // namespace library
