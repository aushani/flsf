#pragma once

#include <memory>

#include <boost/filesystem.hpp>

#include "library/ray_tracing/occ_grid.h"
#include "library/kitti/velodyne_scan.h"

#include "library/flow/flow_image.h"

namespace fs = boost::filesystem;
namespace rt = library::ray_tracing;
namespace kt = library::kitti;

namespace library {
namespace flow {

// Forward declaration
typedef struct DeviceData DeviceData;

class FlowProcessor {
 public:
  FlowProcessor(const fs::path &data_path);
  ~FlowProcessor();

  void Initialize(const kt::VelodyneScan &scan);
  void Update(const kt::VelodyneScan &scan);

  rt::OccGrid GetLastOccGrid() const;
  FlowImage GetFlowImage() const;

 private:
  static constexpr int kMaxVelodyneScanPoints = 150000;
  static constexpr float kResolution = 0.3;
  static constexpr float kMaxRange = 100.0;

  std::unique_ptr<DeviceData> data_;
};

} // namespace tf
} // namespace library
