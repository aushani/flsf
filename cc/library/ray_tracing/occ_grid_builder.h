// Adapted from dascar
#pragma once

#include <memory>

#include <Eigen/Core>

#include "library/ray_tracing/occ_grid.h"

namespace library {
namespace ray_tracing {

// Forward declaration
typedef struct DeviceData DeviceData;

// This class builds occupancy grids from given observations. Upon contruction,
// tt allocates resoucres on the GPU device.
class OccGridBuilder {
 public:
  // max_observations is the maximum number of ray tracing observations (i.e., points
  // from a velodyne scanner) that can be processed at a time. resolution determines the
  // resolution of the occpuancy grid generated. max_range is the maximum distance that
  // a ray tracing observation will sweep out free space in the occupancy grid.
  OccGridBuilder(int max_observations, float resolution, float max_range);
  ~OccGridBuilder();

  // Assumes that the hit origins are at (0, 0, 0)
  OccGrid GenerateOccGrid(const std::vector<Eigen::Vector3f> &hits);

 private:
  const float resolution_;
  const int max_observations_;

  std::unique_ptr<DeviceData> device_data_;

  size_t ProcessData(const std::vector<Eigen::Vector3f> &hits);
};

} // namespace ray_tracing
} // namespace library
