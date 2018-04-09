#pragma once

#include <Eigen/Core>

#include "library/kitti/velodyne_scan.h"
#include "library/kitti/object_label.h"

namespace library {
namespace kitti {

class KittiChallengeData {
 public:
  static KittiChallengeData LoadFrame(const std::string &dirname, int frame);

  const VelodyneScan& GetScan() const;
  const ObjectLabels& GetLabels() const;
  const Eigen::Matrix4d& GetTcv() const;

  bool InCameraView(double x, double y, double z) const;
  Eigen::Vector2d ToCameraPixels(double x, double y, double z) const;

 private:
  struct Calib {
    Eigen::Matrix<double, 3, 4> p;
    Eigen::Matrix4d r_rect;
    Eigen::Matrix4d t_cv;

    Calib(const Eigen::Matrix<double, 3, 4> &pp, const Eigen::Matrix4d &rr, const Eigen::Matrix4d &tt) :
      p(pp), r_rect(rr), t_cv(tt) {}
  };

  VelodyneScan scan_;
  ObjectLabels labels_;
  Calib calib_;

  KittiChallengeData(const VelodyneScan &scan, const ObjectLabels &labels, const Calib &calib);

  static VelodyneScan LoadVelodyneScan(const std::string &dirname, int frame_num);
  static ObjectLabels LoadObjectLabels(const std::string &dirname, int frame_num);
  static Calib LoadCalib(const std::string &dirname, int frame_num);
};

} // namespace kitti
} // namespace library
