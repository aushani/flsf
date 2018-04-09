#include "library/gpu_util/gpu_memory.h"

#include <boost/assert.hpp>

namespace library {
namespace gpu_util {

std::tuple<size_t, size_t> GetGpuMemory() {
  size_t free_bytes=0, total_bytes=0;

  cudaError_t err  = cudaMemGetInfo(&free_bytes, &total_bytes);
  BOOST_ASSERT(err == cudaSuccess);

  return std::make_tuple(free_bytes, total_bytes);
}

} // gpu_util
} // library
