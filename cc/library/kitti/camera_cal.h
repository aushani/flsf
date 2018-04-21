#pragma once

#include <iostream>
#include <string>

#include <boost/filesystem.hpp>
#include <Eigen/Core>

namespace fs = boost::filesystem;

namespace library {
namespace kitti {

class CameraCal {
 public:
  CameraCal(const fs::path &dirname);

  bool InCameraView(double x, double y, double z) const;

 private:
  Eigen::MatrixXd R_rect_;
  Eigen::MatrixXd P_rect_;
  Eigen::MatrixXd T_cv_;

  Eigen::MatrixXd M_;

  void LoadIntrinsics(FILE *fp);
  void LoadExtrinsics(FILE *fp);
};

} // namespace kitti
} // namespace library
