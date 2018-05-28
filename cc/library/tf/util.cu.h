#pragma once

#include <cudnn.h>

#include <boost/assert.hpp>

namespace library {
namespace tf {

cudnnHandle_t cudnn_handle;
bool handle_initialized = false;

cudnnHandle_t GetCudnnHandle() {
  if (!handle_initialized) {
    // Initalize cudnn
    cudnnStatus_t status = cudnnCreate(&cudnn_handle);
    BOOST_ASSERT(status == CUDNN_STATUS_SUCCESS);

    printf("Initialized cudnn\n");
    handle_initialized = true;
  }

  return cudnn_handle;
}

} // namespace tf
} // namespace library
