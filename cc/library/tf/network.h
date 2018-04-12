#pragma once

#include <memory>
#include <string>
#include <vector>

#include <boost/filesystem.hpp>

#include "library/ray_tracing/occ_grid.h"

namespace fs = boost::filesystem;
namespace rt = library::ray_tracing;

namespace library {
namespace tf {

// Forward declaration of data
typedef struct NetworkData NetworkData;

class Network {
 public:
  static std::shared_ptr<Network> LoadNetwork(const fs::path &path);

  void SetInput(const rt::OccGrid &og);

  void Apply();

 private:
  std::unique_ptr<NetworkData> data_;

  Network(NetworkData *data);

  static std::vector<float> LoadFile(const fs::path &path);
};

} // namespace tf
} // namespace library
