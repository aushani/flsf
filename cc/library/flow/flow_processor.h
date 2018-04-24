#pragma once

#include <memory>

#include <boost/filesystem.hpp>

#include "library/ray_tracing/occ_grid.h"
#include "library/kitti/velodyne_scan.h"

#include "library/flow/flow_image.h"
#include "library/flow/classification_map.h"
#include "library/flow/distance_map.h"

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
  FlowProcessor(const FlowProcessor &fp);
  ~FlowProcessor();

  FlowProcessor operator=(const FlowProcessor &fp);

  void Initialize(const kt::VelodyneScan &scan);
  void Update(const kt::VelodyneScan &scan);

  rt::OccGrid GetLastOccGrid1() const;
  rt::OccGrid GetLastOccGrid2() const;

  FlowImage GetFlowImage() const;
  const ClassificationMap& GetClassificationMap() const;
  const DistanceMap& GetDistanceMap() const;

 private:
  std::shared_ptr<DeviceData> data_;

  void UpdateClassificationMap();
  void UpdateDistanceMap();
};

} // namespace tf
} // namespace library
