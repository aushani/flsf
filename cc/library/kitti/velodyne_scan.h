#pragma once

#include <vector>
#include <string>

#include <boost/filesystem.hpp>
#include <Eigen/Core>

namespace fs = boost::filesystem;

namespace library {
namespace kitti {

class VelodyneScan {
 public:
  VelodyneScan(const fs::path &fn);

  const std::vector<Eigen::Vector3f>& GetHits() const;
  const std::vector<float>& GetIntensities() const;

  static std::vector<VelodyneScan> LoadDirectory(const fs::path &dir);

 private:
  std::vector<Eigen::Vector3f> hits_;
  std::vector<float> intensity_;

  VelodyneScan();
};

} // namespace kitti
} // namespace library
