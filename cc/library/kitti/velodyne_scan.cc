#include "library/kitti/velodyne_scan.h"

#include <fstream>

namespace library {
namespace kitti {

VelodyneScan::VelodyneScan(const std::string &fn) {
  std::ifstream input(fn, std::ios::binary);

  float data[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  while (input.good() && !input.eof()) {
    if (!input.read(reinterpret_cast<char*>(&data[0]), 4*sizeof(float))) {
      break;
    }

    hits_.emplace_back(data[0], data[1], data[2]);
    intensity_.push_back(data[3]);
  }
}

const std::vector<Eigen::Vector3f>& VelodyneScan::GetHits() const {
  return hits_;
}

const std::vector<float>& VelodyneScan::GetIntensities() const {
  return intensity_;
}

} // namespace kitti
} // namespace library
