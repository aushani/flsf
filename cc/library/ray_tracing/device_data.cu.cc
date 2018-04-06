#include "library/ray_tracing/device_data.cu.h"

#include <boost/assert.hpp>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>

#include "library/timer/timer.h"

namespace library {
namespace ray_tracing {

DeviceData::DeviceData(float resolution, float max_range, int max_observations) :
 resolution(resolution),
 max_voxel_visits_per_ray(max_range / resolution),
 steps_per_ray(max_voxel_visits_per_ray),
 hits(max_observations),
 locations(max_observations * max_voxel_visits_per_ray),
 log_odds_updates(max_observations * max_voxel_visits_per_ray),
 locations_reduced(max_observations * max_voxel_visits_per_ray),
 log_odds_updates_reduced(max_observations * max_voxel_visits_per_ray) {
}

void DeviceData::CopyData(const std::vector<Eigen::Vector3f> &h) {
  hits.SetPartialData(h);
}

__global__ void RayTracingKernel(DeviceData data) {
  // Figure out which hit this thread is processing
  const int bidx = blockIdx.x;
  const int tidx = threadIdx.x;
  const int threads = blockDim.x;

  const int hit_idx = tidx + bidx * threads;
  if (hit_idx >= data.num_observations) {
    return;
  }

  // Get origin and hit relative to pose
  /*
  float hx = data.hit_x[hit_idx] - data.pose_xyz[0];
  float hy = data.hit_y[hit_idx] - data.pose_xyz[1];
  float hz = data.hit_z[hit_idx] - data.pose_xyz[2];

  float st = sin(data.pose_theta);
  float ct = cos(data.pose_theta);

  float hx_p = ct * hx - st * hy;
  float hy_p = st * hx + ct * hy;
  float hz_p = hz;

  float ox = -data.pose_xyz[0];
  float oy = -data.pose_xyz[1];
  float oz = -data.pose_xyz[2];

  float ox_p = ct * ox - st * oy;
  float oy_p = st * ox + ct * oy;
  float oz_p = oz;

  float hit[3] = {hx_p, hy_p, hz_p};
  float origin[3] = {ox_p, oy_p, oz_p};
  */

  const auto &hit_vector = data.hits(hit_idx);

  float hit[3] = {hit_vector.x(), hit_vector.y(), hit_vector.z()};
  float origin[3] = {0, 0, 0};

  // The following is an implementation of Bresenham's line algorithm to sweep out the ray from the origin of the ray to
  // the hit point.
  // https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
  float ad[3] = {0.0f, 0.0f, 0.0f};  // absolute value of diff * 2
  int sgn[3] = {0, 0, 0};            // which way am i going
  int cur_loc[3] = {0, 0, 0};        // what location am i currently at (starts at origin of ray)
  int end_loc[3] = {0, 0, 0};        // what location am i ending at (ends at the hit)
  int dominant_dim = 0;              // which dim am i stepping through

  for (int i = 0; i < 3; ++i) {
    cur_loc[i] = round(origin[i] / data.resolution);
    end_loc[i] = round(hit[i] / data.resolution);

    ad[i] = fabsf(end_loc[i] - cur_loc[i]) * 2;

    sgn[i] = (end_loc[i] > cur_loc[i]) - (cur_loc[i] > end_loc[i]);

    if (ad[i] > ad[dominant_dim]) {
      dominant_dim = i;
    }
  }

  float err[3];
  for (int i = 0; i < 3; i++) {
    err[i] = ad[i] - ad[dominant_dim] / 2;
  }

  // walk down ray
  size_t mem_step_size = data.num_observations;
  size_t mem_idx = hit_idx;
  bool valid = true;
  for (int step = 0; step < (data.steps_per_ray - 1); ++step) {
    Location loc(cur_loc[0], cur_loc[1], cur_loc[2]);
    float loUpdate = data.kLogOddsUnknown;

    // Are we done? Have we reached the hit point?
    // Don't quit the loop just yet. We need to 0 out the rest of log odds updates.
    if ((sgn[dominant_dim] > 0) ? (cur_loc[dominant_dim] >= end_loc[dominant_dim])
                                : (cur_loc[dominant_dim] <= end_loc[dominant_dim])) {
      valid = false;
    }

    if (valid) {
      // step forwards
      for (int dim = 0; dim < 3; ++dim) {
        if (dim != dominant_dim) {
          if (err[dim] >= 0) {
            cur_loc[dim] += sgn[dim];
            err[dim] -= ad[dominant_dim];
          }
        }
      }

      for (int dim = 0; dim < 3; ++dim) {
        if (dim == dominant_dim) {
          cur_loc[dim] += sgn[dim];
        } else {
          err[dim] += ad[dim];
        }
      }

      loUpdate = data.kLogOddsFree;

      // Are we in or out of range?
      if (data.oor_valid && data.oor(loc)) {
        step--;
        continue;
      }
    }

    // Now write out key value pair
    data.locations(mem_idx) = loc;
    data.log_odds_updates(mem_idx) = loUpdate;

    mem_idx += mem_step_size;
  }

  Location loc(end_loc[0], end_loc[1], end_loc[2]);

  bool valid_end_point = (!data.oor_valid || !data.oor(loc));

  // Now write out key value pair
  if (valid_end_point) {
    data.locations(mem_idx) = loc;
    data.log_odds_updates(mem_idx) = data.kLogOddsOccupied;
  } else {
    data.locations(mem_idx) = loc;
    data.log_odds_updates(mem_idx) = data.kLogOddsUnknown;
  }
}

struct NoUpdate {
  __host__ __device__ bool operator()(const float x) const { return fabs(x) < 1e-6; }
};

void DeviceData::RunKernel() {
  //library::timer::Timer t;

  //t.Start();
  int blocks = ceil(static_cast<float>(num_observations) / kThreadsPerBlock);
  RayTracingKernel<<<blocks, kThreadsPerBlock>>>(*this);
  cudaError_t err = cudaDeviceSynchronize();
  BOOST_ASSERT(err == cudaSuccess);
  //printf("\tTook %5.3f ms to run kernel\n", t.GetMs());
}

size_t DeviceData::ReduceLogOdds() {
  // Accumulate all the updates
  size_t num_updates = num_observations * steps_per_ray;

  // First prune unnecessary updates
  //t.Start();
  //thrust::device_ptr<Location> dp_locations(locations);
  //thrust::device_ptr<float> dp_updates(log_odds_updates);

  auto dp_locations_end = thrust::remove_if(locations.Begin(), locations.Begin() + num_updates, log_odds_updates.Begin(), NoUpdate());
  auto dp_updates_end = thrust::remove_if(log_odds_updates.Begin(), log_odds_updates.Begin() + num_updates, NoUpdate());

  //auto dp_locations_end = thrust::remove_if(dp_locations, dp_locations + num_updates, dp_updates, NoUpdate());
  //auto dp_updates_end = thrust::remove_if(dp_updates, dp_updates + num_updates, NoUpdate());
  //printf("\tTook %5.3f to prune from %ld to %ld\n", t.GetMs(), num_updates, dp_locations_end - dp_locations);
  //num_updates = dp_locations_end - dp_locations;
  num_updates = dp_locations_end - locations.Begin();

  // Now reduce updates to resulting log odds
  //t.Start();
  //thrust::sort_by_key(dp_locations, dp_locations + num_updates, dp_updates);
  thrust::sort_by_key(locations.Begin(), locations.Begin() + num_updates, log_odds_updates.Begin());
  //printf("\tTook %5.3f to sort\n", t.GetMs());

  //t.Start();
  //thrust::device_ptr<Location> dp_locs_reduced(locations_reduced);
  //thrust::device_ptr<float> dp_lo_reduced(log_odds_updates_reduced);

  //thrust::pair<thrust::device_ptr<Location>, thrust::device_ptr<float> > new_ends = thrust::reduce_by_key(
  //    dp_locations, dp_locations + num_updates, dp_updates, dp_locs_reduced, dp_lo_reduced);
  //num_updates = new_ends.first - dp_locs_reduced;
  auto new_ends = thrust::reduce_by_key(locations.Begin(), locations.Begin() + num_updates, log_odds_updates.Begin(),
      locations_reduced.Begin(), log_odds_updates_reduced.Begin());
  num_updates = new_ends.first - locations_reduced.Begin();
  //printf("\tTook %5.3f to reduce\n", t.GetMs());

  return num_updates;
}

}  // namespace ray_tracing
}  // namespace library
