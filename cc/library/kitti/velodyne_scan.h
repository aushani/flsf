#pragma once

#include <vector>
#include <string>

#include <Eigen/Core>

namespace library {
namespace kitti {

class VelodyneScan {
 public:
  VelodyneScan(const std::string &fn);

  const std::vector<Eigen::Vector3f>& GetHits() const;
  const std::vector<float>& GetIntensities() const;

 private:
  std::vector<Eigen::Vector3f> hits_;
  std::vector<float> intensity_;

};

} // namespace kitti
} // namespace library
