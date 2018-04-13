#include "library/flow/flow_processor.h"

namespace library {
namespace flow {

struct DeviceData {
  DeviceData() {

  };

  int test;
};

FlowProcessor::FlowProcessor() :
 data_(new DeviceData()) {
}

FlowProcessor::~FlowProcessor() {
}

void FlowProcessor::Initialize(const kt::VelodyneScan &scan) {
  printf("TODO\n");
}

void FlowProcessor::Update(const kt::VelodyneScan &scan) {
  printf("TODO\n");
}

} // namespace tf
} // namespace library

