#pragma once

#include <memory>

#include "library/ray_tracing/occ_grid.h"
#include "library/kitti/velodyne_scan.h"

namespace rt = library::ray_tracing;
namespace kt = library::kitti;

namespace library {
namespace flow {

// Forward declaration
typedef struct DeviceData DeviceData;

class FlowProcessor {
 public:
  FlowProcessor();
  ~FlowProcessor();

  void Initialize(const kt::VelodyneScan &scan);
  void Update(const kt::VelodyneScan &scan);

 private:
  static constexpr int kMaxVelodyneScanPoints = 150000;
  static constexpr float kResolution = 0.3;
  static constexpr float kMaxRange = 100.0;

  std::unique_ptr<DeviceData> data_;
};

} // namespace tf
} // namespace library
