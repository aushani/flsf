#include "library/flow/flow_processor.h"

#include <boost/optional.hpp>

#include "library/ray_tracing/occ_grid_builder.h"
#include "library/tf/network.cu.h"
#include "library/gpu_util/gpu_data.cu.h"
#include "library/gpu_util/host_data.cu.h"

#include "library/flow/metric_distance.cu.h"
#include "library/flow/solver.cu.h"

namespace tf = library::tf;

namespace library {
namespace flow {

struct DeviceData {
  DeviceData(int max_obs, float res, float max_range, const fs::path &data_path) :
   og_builder(max_obs, res, max_range),
   network(tf::Network::LoadNetwork(data_path)),
   distance_computer(),
   solver(),
   last_encoding(167, 167, 25),     // XXX magic numbers
   distance(167, 167, 31, 31),      // XXX magic numbers
   raw_flow(167, 167, 2),           // XXX magic numbers
   classification_map(167, 167) {   // XXX magic numbers
  };

  rt::OccGridBuilder og_builder;
  tf::Network        network;
  MetricDistance     distance_computer;
  Solver             solver;

  boost::optional<rt::OccGrid>              last_og;
  gu::GpuData<3, float>                     last_encoding;

  gu::GpuData<4, float>                     distance;
  gu::GpuData<3, int>                       raw_flow;
  boost::optional<FlowImage>                flow_image;
  ClassificationMap                         classification_map;
};

FlowProcessor::FlowProcessor(const fs::path &data_path) :
 data_(new DeviceData(kMaxVelodyneScanPoints, kResolution, kMaxRange, data_path)) {
}

FlowProcessor::~FlowProcessor() {
}

void FlowProcessor::Initialize(const kt::VelodyneScan &scan) {
  auto og = data_->og_builder.GenerateOccGrid(scan.GetHits());
  data_->last_og = og;

  data_->network.SetInput(og);
  data_->network.Apply();
  data_->last_encoding.CopyFrom(data_->network.GetEncoding());
}

void FlowProcessor::Update(const kt::VelodyneScan &scan) {
  // Get occ grid
  auto og = data_->og_builder.GenerateOccGrid(scan.GetHits());

  // Run encoding network
  data_->network.SetInput(og);
  data_->network.Apply();

  // Get encoding
  auto encoding = data_->network.GetEncoding();
  auto classification = data_->network.GetClassification();

  // Get distance
  data_->distance_computer.ComputeDistance(data_->last_encoding, encoding, &data_->distance);

  // Compute flow
  auto fi = data_->solver.ComputeFlow(data_->distance, classification, &data_->raw_flow);

  // Update cached state
  data_->last_og = og;
  data_->last_encoding.CopyFrom(encoding);
  data_->flow_image = fi;

  gu::HostData<3, float> hd(classification);

  for (int i=0; i<hd.GetDim(0); i++) {
    for (int j=0; j<hd.GetDim(1); j++) {
      for (int k=0; k<hd.GetDim(2); k++) {
        int ii = data_->classification_map.MinX() + i;
        int jj = data_->classification_map.MinY() + j;
        ObjectClass c = ClassificationMap::IntToObjectClass(k);

        data_->classification_map.SetClassScore(ii, jj, c, hd(i, j, k));
      }
    }
  }
}

rt::OccGrid FlowProcessor::GetLastOccGrid() const {
  BOOST_ASSERT(data_->last_og);

  return *data_->last_og;
}

FlowImage FlowProcessor::GetFlowImage() const {
  BOOST_ASSERT(data_->flow_image);
  return *data_->flow_image;
}

const ClassificationMap& FlowProcessor::GetClassificationMap() const {
  return data_->classification_map;
}

} // namespace tf
} // namespace library
