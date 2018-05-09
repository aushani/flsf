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
   solver(167, 167, 31),                  // XXX magic numbers
   last_encoding1(167, 167, 10),          // XXX magic numbers
   last_encoding2(167, 167, 10),          // XXX magic numbers
   distance(167, 167, 31, 31),            // XXX magic numbers
   filter(167, 167, 2),                   // XXX magic numbers
   //classification_map(167, 167, res),   // XXX magic numbers
   filter_map(167, 167, res),             // XXX magic numbers
   distance_map(167, 167, 31, res) {      // XXX magic numbers
    last_encoding1.SetCoalesceDim(0);
    last_encoding2.SetCoalesceDim(0);

    filter.SetCoalesceDim(0);
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

  gu::GpuData<3, float>                     filter;

  boost::optional<FlowImage>                flow_image;
  //ClassificationMap                         classification_map;
  FilterMap                                 filter_map;
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

void FlowProcessor::Initialize(const kt::VelodyneScan &scan) {
  auto og = data_->og_builder.GenerateOccGrid(scan.GetHits());
  data_->last_og2 = og;

  data_->network.SetInput(og);
  data_->network.Apply();
  data_->last_encoding2.CopyFrom(data_->network.GetEncoding());
}

void FlowProcessor::Update(const kt::VelodyneScan &scan) {
  library::timer::Timer t;

  // Get occ grid
  t.Start();
  auto og = data_->og_builder.GenerateOccGrid(scan.GetHits());
  printf("Took %5.3f ms to generate occ grid\n", t.GetMs());

  // Run encoding network
  t.Start();
  data_->network.SetInput(og);
  printf("Took %5.3f ms to set occ grid input to network\n", t.GetMs());

  t.Start();
  data_->network.Apply();
  printf("Took %5.3f ms to apply network\n", t.GetMs());

  // Get encoding
  const auto &encoding = data_->network.GetEncoding();
  //const auto &classification = data_->network.GetClassification();
  const auto &filter = data_->network.GetFilter();

  // Get distance
  data_->distance_computer.ComputeDistance(data_->last_encoding2, encoding, &data_->distance);

  // Compute flow
  auto fi = data_->solver.ComputeFlow(data_->distance, filter, data_->resolution, iterations_);

  // Update cached state
  data_->last_og1 = data_->last_og2;
  data_->last_og2 = og;

  data_->filter.CopyFrom(filter);

  auto tmp = data_->last_encoding1;
  data_->last_encoding1 = data_->last_encoding2;
  data_->last_encoding2 = tmp;

  data_->last_encoding2.CopyFrom(encoding);
  data_->flow_image = fi;

  //UpdateClassificationMap();
  UpdateFilterMap();
  UpdateDistanceMap();
}

void FlowProcessor::Refresh() {
  // Get distance
  data_->distance_computer.ComputeDistance(data_->last_encoding1, data_->last_encoding2, &data_->distance);

  // Compute flow
  auto fi = data_->solver.ComputeFlow(data_->distance, data_->filter, data_->resolution, iterations_);

  data_->flow_image = fi;

  UpdateFilterMap();
  UpdateDistanceMap();
}

//void FlowProcessor::UpdateClassificationMap() {
//  const auto &classification = data_->network.GetClassification();
//
//  gu::HostData<3, float> hd(classification);
//
//  for (int i=0; i<hd.GetDim(0); i++) {
//    for (int j=0; j<hd.GetDim(1); j++) {
//      for (int k=0; k<hd.GetDim(2); k++) {
//        int ii = data_->classification_map.MinX() + i;
//        int jj = data_->classification_map.MinY() + j;
//        kt::ObjectClass c = kt::IntToObjectClass(k);
//
//        data_->classification_map.SetClassScore(ii, jj, c, hd(i, j, k));
//      }
//    }
//  }
//}

void FlowProcessor::UpdateFilterMap() {
  gu::HostData<3, float> hd(data_->filter);

  for (int i=0; i<hd.GetDim(0); i++) {
    for (int j=0; j<hd.GetDim(1); j++) {
      float x1 = hd(i, j, 0);
      float x2 = hd(i, j, 1);

      float denom = std::exp(x1) + std::exp(x2);
      float prob = std::exp(x1) / denom;

      int ii = data_->filter_map.MinX() + i;
      int jj = data_->filter_map.MinY() + j;

      data_->filter_map.SetFilterProbability(ii, jj, prob);
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

//const ClassificationMap& FlowProcessor::GetClassificationMap() const {
//  return data_->classification_map;
//}

const FilterMap& FlowProcessor::GetFilterMap() const {
  return data_->filter_map;
}

const DistanceMap& FlowProcessor::GetDistanceMap() const {
  return data_->distance_map;
}

} // namespace tf
} // namespace library
