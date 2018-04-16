#include "library/flow/flow_processor.h"

#include <boost/optional.hpp>

#include "library/ray_tracing/occ_grid_builder.h"
#include "library/tf/network.cu.h"
#include "library/flow/metric_distance.cu.h"
#include "library/flow/solver.cu.h"

namespace tf = library::tf;

namespace library {
namespace flow {

struct DeviceData {
  DeviceData(int max_obs, float res, float max_range, const fs::path &data_path) :
   og_builder_(max_obs, res, max_range),
   network_(tf::Network::LoadNetwork(data_path)),
   distance_computer_(),
   solver_(),
   last_encoding_(167, 167, 25), // XXX magic numbers
   distance_(167, 167, 31, 31),  // XXX magic numbers
   raw_flow_(167, 167, 2) {      // XXX magic numbers
  };

  rt::OccGridBuilder og_builder_;
  tf::Network        network_;
  MetricDistance     distance_computer_;
  Solver             solver_;

  boost::optional<rt::OccGrid>              last_og_;
  gu::GpuData<3, float>                     last_encoding_;

  gu::GpuData<4, float>                     distance_;
  gu::GpuData<3, int>                       raw_flow_;
  boost::optional<FlowImage>                flow_image_;
};

FlowProcessor::FlowProcessor(const fs::path &data_path) :
 data_(new DeviceData(kMaxVelodyneScanPoints, kResolution, kMaxRange, data_path)) {
}

FlowProcessor::~FlowProcessor() {
}

void FlowProcessor::Initialize(const kt::VelodyneScan &scan) {
  auto og = data_->og_builder_.GenerateOccGrid(scan.GetHits());
  data_->last_og_ = og;

  data_->network_.SetInput(og);
  data_->network_.Apply();
  data_->last_encoding_.CopyFrom(data_->network_.GetEncoding());
}

void FlowProcessor::Update(const kt::VelodyneScan &scan) {
  // Get occ grid
  auto og = data_->og_builder_.GenerateOccGrid(scan.GetHits());

  // Run encoding network
  data_->network_.SetInput(og);
  data_->network_.Apply();

  // Get encoding
  auto encoding = data_->network_.GetEncoding();

  // Get distance
  data_->distance_computer_.ComputeDistance(data_->last_encoding_, encoding, &data_->distance_);

  // Compute flow
  auto fi = data_->solver_.ComputeFlow(data_->distance_, &data_->raw_flow_);

  // Update cached state
  data_->last_og_ = og;
  data_->last_encoding_.CopyFrom(encoding);
  data_->flow_image_ = fi;
}

rt::OccGrid FlowProcessor::GetLastOccGrid() const {
  BOOST_ASSERT(data_->last_og_);

  return *data_->last_og_;
}

FlowImage FlowProcessor::GetFlowImage() const {
  BOOST_ASSERT(data_->flow_image_);
  return *data_->flow_image_;
}

} // namespace tf
} // namespace library
