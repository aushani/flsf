#include "library/kitti/velodyne_scan.h"

#include <fstream>
#include <thread>

#include <boost/assert.hpp>
#include <boost/format.hpp>

namespace library {
namespace kitti {

VelodyneScan::VelodyneScan() {
}

VelodyneScan::VelodyneScan(const fs::path &fn) {
  std::ifstream input(fn.string(), std::ios::binary);

  input.seekg(0, input.end);
  int sz = input.tellg();
  input.seekg(0, input.beg);

  int n_floats = sz / (sizeof(float));

  // Read all floats
  std::vector<float> data(n_floats, 0);
  input.read(reinterpret_cast<char*>(data.data()), sz);
  BOOST_ASSERT(input);

  // Reserve space
  hits_.reserve(data.size() / 4);
  intensity_.reserve(data.size() / 4);

  // Populate
  for (size_t i=0; i<data.size(); i+=4) {
    float x = data[i + 0];
    float y = data[i + 1];
    float z = data[i + 2];
    float intens = data[i + 3];

    hits_.emplace_back(x, y, z);
    intensity_.push_back(intens);
  }
}

const std::vector<Eigen::Vector3f>& VelodyneScan::GetHits() const {
  return hits_;
}

const std::vector<float>& VelodyneScan::GetIntensities() const {
  return intensity_;
}

std::vector<VelodyneScan> VelodyneScan::LoadDirectory(const fs::path &dir) {
  // How many files are there?
  size_t frame_num = 0;
  std::vector<fs::path> paths;
  while (true) {
    std::string fn = (boost::format("%010d.bin") % frame_num).str();
    fs::path path = dir / fn;

    if (!fs::exists(path)) {
      break;
    }

    frame_num++;
    paths.push_back(path);
  }

  // Reserve space
  std::vector<VelodyneScan> res(paths.size(), VelodyneScan());

  // Now load all (multi threaded)
  std::vector<std::thread> load_threads;
  for (size_t i = 0; i<frame_num; i++) {
    const auto &path = paths[i];
    load_threads.emplace_back( [&res, i, &path] {res[i] = VelodyneScan(path);} );
  }

  // Wait for all threads to finish
  for (auto &t : load_threads) {
    t.join();
  }

  return res;
}

} // namespace kitti
} // namespace library
