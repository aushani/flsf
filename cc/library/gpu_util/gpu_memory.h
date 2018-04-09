#pragma once

#include <tuple>

namespace library {
namespace gpu_util {

std::tuple<size_t, size_t> GetGpuMemory();

} // gpu_util
} // library
