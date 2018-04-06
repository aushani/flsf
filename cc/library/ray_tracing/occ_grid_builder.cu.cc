// Adapted from dascar
#include "library/ray_tracing/occ_grid_builder.h"

#include <iostream>

#include <boost/assert.hpp>

#include <thrust/device_ptr.h>

#include "library/ray_tracing/occ_grid_location.h"
#include "library/ray_tracing/device_data.cu.h"
#include "library/timer/timer.h"

namespace tr = library::timer;

namespace library {
namespace ray_tracing {

OccGridBuilder::OccGridBuilder(int max_observations, float resolution, float max_range)
    : max_observations_(max_observations),
      resolution_(resolution),
      device_data_(new DeviceData(resolution, max_range, max_observations)) {
}

OccGridBuilder::~OccGridBuilder() {
  device_data_->FreeDeviceMemory();
}

size_t OccGridBuilder::ProcessData(const std::vector<Eigen::Vector3f> &hits) {
  // First, we need to send the data to the GPU device
  device_data_->CopyData(hits);

  // Now run ray tracing on the GPU device
  device_data_->RunKernel();
  return device_data_->ReduceLogOdds();
}

OccGrid OccGridBuilder::GenerateOccGrid(const std::vector<Eigen::Vector3f> &hits) {
  //library::timer::Timer t;

  BOOST_ASSERT(hits.size() <= max_observations_);

  // Check for empty data
  if (hits.size() == 0) {
    std::vector<Location> location_vector;
    std::vector<float> lo_vector;
    return OccGrid(location_vector, lo_vector, resolution_);
  }

  size_t num_updates = ProcessData(hits);

  // Copy result from GPU device to host
  //t.Start();
  std::vector<Location> location_vector(num_updates);
  std::vector<float> lo_vector(num_updates);
  cudaMemcpy(location_vector.data(), device_data_->locations_reduced.GetDevicePointer(), sizeof(Location) * num_updates, cudaMemcpyDeviceToHost);
  cudaMemcpy(lo_vector.data(), device_data_->log_odds_updates_reduced.GetDevicePointer(), sizeof(float) * num_updates, cudaMemcpyDeviceToHost);
  cudaError_t err = cudaDeviceSynchronize();
  BOOST_ASSERT(err == cudaSuccess);
  //printf("\tTook %5.3f to copy to host\n", t.GetMs());

  return OccGrid(location_vector, lo_vector, resolution_);
}

}  // namespace ray_tracing
}  // namespace library
