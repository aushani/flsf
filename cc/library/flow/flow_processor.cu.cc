#include "library/flow/flow_processor.h"

#include <boost/optional.hpp>

#include "library/ray_tracing/occ_grid_builder.h"
#include "library/tf/network.cu.h"
#include "library/gpu_util/gpu_data.cu.h"
#include "library/gpu_util/host_data.cu.h"
#include "library/kitti/object_class.h"
#include "library/params/params.h"
#include "library/timer/timer.h"

#include "library/flow/metric_distance.cu.h"
#include "library/flow/solver.cu.h"

namespace ps = library::params;
namespace tf = library::tf;

namespace library {
namespace flow {

struct DeviceData {
  DeviceData(int max_obs, float res, float max_range, const fs::path &data_path) :
   resolution(res),
   og_builder(max_obs, res, max_range),
   network(tf::Network::LoadNetwork(data_path)),
   distance_computer(),
   solver(167, 167, 31),                                  // XXX magic numbers
   last_encoding1(167, 167, network.GetEncodingDim()),    // XXX magic numbers
   last_encoding2(167, 167, network.GetEncodingDim()),    // XXX magic numbers
   distance(167, 167, 31, 31),                            // XXX magic numbers
   p_background1(167, 167),                               // XXX magic numbers
   p_background2(167, 167),                               // XXX magic numbers
   filter_map1(167, 167, res),                            // XXX magic numbers
   filter_map2(167, 167, res),                            // XXX magic numbers
   distance_map(167, 167, 31, res) {                      // XXX magic numbers
  };

  float resolution;

  rt::OccGridBuilder og_builder;
  tf::Network        network;
  MetricDistance     distance_computer;
  Solver             solver;

  boost::optional<rt::OccGrid>              last_og1;   // Flow is from this og
  boost::optional<rt::OccGrid>              last_og2;   // Flow is to this og

  gu::GpuData<3, float>                     last_encoding1;
  gu::GpuData<3, float>                     last_encoding2;

  gu::GpuData<4, float>                     distance;

  gu::GpuData<2, float>                     p_background1;
  gu::GpuData<2, float>                     p_background2;

  boost::optional<FlowImage>                flow_image;
  //ClassificationMap                         classification_map;
  FilterMap                                 filter_map1;
  FilterMap                                 filter_map2;
  DistanceMap                               distance_map;
};

FlowProcessor::FlowProcessor(const fs::path &data_path) :
 data_(std::make_shared<DeviceData>(ps::kMaxVelodyneScanPoints,
                                    ps::kResolution,
                                    ps::kMaxRange,
                                    data_path)) {
}

FlowProcessor::FlowProcessor(const FlowProcessor &fp) :
 data_(fp.data_) {
}

FlowProcessor::~FlowProcessor() {
}

FlowProcessor FlowProcessor::operator=(const FlowProcessor &fp) {
  data_ = fp.data_;

  return *this;
}

void FlowProcessor::SetIterations(int iters) {
  iterations_ = iters;
}

int FlowProcessor::GetIterations() const {
  return iterations_;
}

void FlowProcessor::SetSmoothing(float val) {
  data_->solver.SetSmoothing(val);
}

void FlowProcessor::Initialize(const kt::VelodyneScan &scan) {
  auto og = data_->og_builder.GenerateOccGrid(scan.GetHits());
  data_->last_og2 = og;

  data_->network.SetInput(og);
  data_->network.Apply(&data_->last_encoding2, &data_->p_background2);
}

void FlowProcessor::Update(const kt::VelodyneScan &scan, bool copy_data) {
  library::timer::Timer t;

  gu::GpuData<3, float> prev_encoding     = data_->last_encoding2;
  gu::GpuData<3, float> encoding          = data_->last_encoding1;

  gu::GpuData<2, float> prev_p_background = data_->p_background2;
  gu::GpuData<2, float> p_background      = data_->p_background1;

  // Get occ grid
  t.Start();
  auto og = data_->og_builder.GenerateOccGrid(scan.GetHits());
  printf("Took %5.3f ms to generate occ grid\n", t.GetMs());

  // Run encoding network
  t.Start();
  data_->network.SetInput(og);
  printf("Took %5.3f ms to set occ grid input to network\n", t.GetMs());

  t.Start();
  data_->network.Apply(&encoding, &p_background);
  printf("Took %5.3f ms to apply network\n", t.GetMs());

  // Get distance
  data_->distance_computer.ComputeDistance(prev_encoding, encoding, &data_->distance);

  // Compute flow
  data_->flow_image = data_->solver.ComputeFlow(data_->distance, prev_p_background,
                                                data_->resolution, iterations_);

  // Update cached state ////////////
  t.Start();

  // Update
  data_->last_og1 = data_->last_og2;
  data_->last_og2 = og;

  // Swap
  data_->last_encoding1 = prev_encoding;
  data_->last_encoding2 = encoding;

  // Swap
  data_->p_background1 = prev_p_background;
  data_->p_background2 = p_background;

  // Copy to host
  if (copy_data) {
    t.Start();
    UpdateBackgroundFilterMap();
    UpdateDistanceMap();
    printf("Took %5.3f ms to copy proccesing data to host\n", t.GetMs());
  }
}

void FlowProcessor::Refresh() {
  // Get distance
  data_->distance_computer.ComputeDistance(data_->last_encoding1, data_->last_encoding2, &data_->distance);

  // Compute flow
  auto fi = data_->solver.ComputeFlow(data_->distance, data_->p_background1,
                                      data_->resolution, iterations_);

  data_->flow_image = fi;

  UpdateBackgroundFilterMap();
  UpdateDistanceMap();
}

void FlowProcessor::UpdateBackgroundFilterMap() {
  gu::HostData<2, float> hd1(data_->p_background1);
  gu::HostData<2, float> hd2(data_->p_background2);

  for (int i=0; i<hd1.GetDim(0); i++) {
    int ii = data_->filter_map1.MinX() + i;

    for (int j=0; j<hd1.GetDim(1); j++) {
      int jj = data_->filter_map1.MinY() + j;

      data_->filter_map1.SetFilterProbability(ii, jj, hd1(i, j));
      data_->filter_map2.SetFilterProbability(ii, jj, hd2(i, j));
    }
  }
}

void FlowProcessor::UpdateDistanceMap() {
  gu::HostData<4, float> hd(data_->distance);

  for (int i=0; i<hd.GetDim(0); i++) {
    int ii = data_->distance_map.MinX() + i;

    for (int j=0; j<hd.GetDim(1); j++) {
      int jj = data_->distance_map.MinY() + j;

      for (int kk=0; kk<hd.GetDim(2); kk++) {
        int di = data_->distance_map.MinOffset() + kk;

        for (int ll=0; ll<hd.GetDim(3); ll++) {
          int dj = data_->distance_map.MinOffset() + ll;

          float dist = hd(i, j, kk, ll);
          data_->distance_map.SetDistance(ii, jj, di, dj, dist);
        }
      }
    }
  }
}

rt::OccGrid FlowProcessor::GetLastOccGrid1() const {
  BOOST_ASSERT(data_->last_og1);

  return *data_->last_og1;
}

rt::OccGrid FlowProcessor::GetLastOccGrid2() const {
  BOOST_ASSERT(data_->last_og2);

  return *data_->last_og2;
}

FlowImage FlowProcessor::GetFlowImage() const {
  BOOST_ASSERT(data_->flow_image);
  return *data_->flow_image;
}

const FilterMap& FlowProcessor::GetBackgroundFilterMap1() const {
  return data_->filter_map1;
}

const FilterMap& FlowProcessor::GetBackgroundFilterMap2() const {
  return data_->filter_map2;
}

const DistanceMap& FlowProcessor::GetDistanceMap() const {
  return data_->distance_map;
}

} // namespace tf
} // namespace library
