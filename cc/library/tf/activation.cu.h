#pragma once

#include <thrust/transform.h>

namespace library {
namespace tf {

enum class Activation {
  NONE,
  LEAKY_RELU,
};

struct leaky_relu : public thrust::unary_function<float, float> {
  __host__ __device__ float operator()(float x) {
    if (x < 0) {
      return x * 0.2;
    }
    return x;
  }
};


} // namespace tf
} // namespace library
