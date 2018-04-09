#include "library/kitti/kitti_challenge_data.h"

namespace library {
namespace kitti {

KittiChallengeData::KittiChallengeData(const VelodyneScan &scan, const ObjectLabels &labels, const Calib &calib) :
 scan_(scan), labels_(labels), calib_(calib) {}

KittiChallengeData KittiChallengeData::LoadFrame(const std::string &dirname, int frame) {
  VelodyneScan vs = LoadVelodyneScan(dirname, frame);
  ObjectLabels ols = LoadObjectLabels(dirname, frame);
  Calib calib = LoadCalib(dirname, frame);

  return KittiChallengeData(vs, ols, calib);
}

VelodyneScan KittiChallengeData::LoadVelodyneScan(const std::string &dirname, int frame_num) {
  char fn[1000];
  sprintf(fn, "%s/data_object_velodyne/training/velodyne/%06d.bin",
      dirname.c_str(), frame_num);

  //printf("Loading velodyne from %s\n", fn);

  return VelodyneScan(fn);
}

ObjectLabels KittiChallengeData::LoadObjectLabels(const std::string &dirname, int frame_num) {
  // Load Labels
  char fn[1000];
  sprintf(fn, "%s/data_object_label_2/training/label_2/%06d.txt",
      dirname.c_str(), frame_num);

  //printf("Loading labels from %s\n", fn);

  ObjectLabels labels = ObjectLabel::Load(fn);

  return labels;
}

KittiChallengeData::Calib KittiChallengeData::LoadCalib(const std::string &dirname, int frame_num) {
  // Load Labels
  char fn[1000];
  sprintf(fn, "%s/data_object_calib/training/calib/%06d.txt",
      dirname.c_str(), frame_num);

  //printf("Loading calib from %s\n", fn);

  Eigen::Matrix<double, 3, 4> p;
  Eigen::Matrix4d r_rect;
  Eigen::Matrix4d t_cv;

  ObjectLabel::LoadCalib(fn, &p, &r_rect, &t_cv);

  return Calib(p, r_rect, t_cv);
}

const VelodyneScan& KittiChallengeData::GetScan() const {
  return scan_;
}

const ObjectLabels& KittiChallengeData::GetLabels() const {
  return labels_;
}

const Eigen::Matrix4d& KittiChallengeData::GetTcv() const {
  return calib_.t_cv;
}

Eigen::Vector2d KittiChallengeData::ToCameraPixels(double x, double y, double z) const {
  Eigen::Vector4d p_x(x, y, z, 1.0);
  Eigen::Vector3d p_c = calib_.p * calib_.r_rect * calib_.t_cv * p_x;

  return p_c.hnormalized();
}

bool KittiChallengeData::InCameraView(double x, double y, double z) const {
  Eigen::Vector4d p_x(x, y, z, 1.0);
  Eigen::Vector3d p_c = calib_.p * calib_.r_rect * calib_.t_cv * p_x;

  if (p_c.z() < 0) {
    return false;
  }

  double x_c = p_c.x() / p_c.z();
  double y_c = p_c.y() / p_c.z();

  if (x_c < 0 || x_c > 1392) {
    return false;
  }

  if (y_c < 0 || y_c > 512) {
    return false;
  }

  return true;
}

} // namespace kitti
} // namespace library
