#pragma once

#include "library/gpu_util/data.cu.h"

namespace library {
namespace gpu_util {

template<int N, class T>
using HostData = Data<N, T, DataLocation::ON_HOST>;

} // namespace gpu_util
} // namespace library
