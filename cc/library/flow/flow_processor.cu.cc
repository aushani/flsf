#include "library/flow/flow_processor.h"

#include <boost/optional.hpp>

#include "library/ray_tracing/occ_grid_builder.h"

namespace library {
namespace flow {

struct DeviceData {
  DeviceData(int max_obs, float res, float max_range) :
   og_builder_(max_obs, res, max_range) {

  };

  rt::OccGridBuilder og_builder_;

  boost::optional<rt::OccGrid> last_og_;
};

FlowProcessor::FlowProcessor() :
 data_(new DeviceData(kMaxVelodyneScanPoints, kResolution, kMaxRange)) {
}

FlowProcessor::~FlowProcessor() {
}

void FlowProcessor::Initialize(const kt::VelodyneScan &scan) {
  printf("TODO\n");

  data_->last_og_ = data_->og_builder_.GenerateOccGrid(scan.GetHits());
}

void FlowProcessor::Update(const kt::VelodyneScan &scan) {
  printf("TODO\n");

  data_->last_og_ = data_->og_builder_.GenerateOccGrid(scan.GetHits());
}

rt::OccGrid FlowProcessor::GetLastOccGrid() const {
  BOOST_ASSERT(data_->last_og_);

  return *data_->last_og_;
}

} // namespace tf
} // namespace library
