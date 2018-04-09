#pragma once

#include <boost/assert.hpp>

#include <vector>
#include <cstring>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace library {
namespace kitti {

struct ObjectLabel;

typedef std::vector<ObjectLabel> ObjectLabels;

struct ObjectLabel {
  enum Type {
    CAR,
    VAN,
    TRUCK,
    PEDESTRIAN,
    PERSON_SITTING,
    CYCLIST,
    TRAM,
    MISC,
    DONT_CARE
  };

  ObjectLabel() {
    H_camera_object.setZero();
  }

  static ObjectLabels Load(const char *fn);
  static void Save(const ObjectLabels &labels, const char *fn);

  static void LoadCalib(const char *fn, Eigen::Matrix<double, 3, 4> *p, Eigen::Matrix4d *r, Eigen::Matrix4d *t_cv);

  Type type = DONT_CARE;            // Describes the type of object
  float truncated = 0;              // float from 0 (non-truncated) to 1 (truncated), how much of object left image boundaries
  float occluded = 3;               // 0 = fully visible, 1 = partly occluded, 2 = largely occluded, 3 = unknown
  float alpha = 0;                  // Observation angle of object, -pi to pi
  float bbox[4] = {0, 0, 0, 0};     // 0-based index of left, top, right, bottom pixel coordinates of bounding box in image plane
  float dimensions[3] = {0, 0, 0};  // 3d height width length in meter
  float location[3] = {0, 0, 0};    // 3d location x y z in camera coordinates in meters
  float rotation_y = 0;             // rotation around y-axis in camera coordinates, -pi to pi
  float score = 0;                  // for results only, float indicated confidence (higher is better)

  // Helper stuff
  Eigen::Matrix4d H_camera_object;

  static Type GetType(const char *type);
  static const char* GetString(const Type &type);

  bool Care() const {
    return type != DONT_CARE;
  }

 private:

  void ComputeTransforms() {
    // Transformations
    Eigen::Affine3d ry(Eigen::AngleAxisd(rotation_y, Eigen::Vector3d(0, 1, 0)));

    double tx = location[0];
    double ty = location[1];
    double tz = location[2];
    Eigen::Affine3d t(Eigen::Translation3d(Eigen::Vector3d(tx, ty, tz)));

    Eigen::Matrix4d H_object_camera = (t*ry).matrix();
    H_camera_object = H_object_camera.inverse();
  }
};

} // namespace kitti
} // namespace library
