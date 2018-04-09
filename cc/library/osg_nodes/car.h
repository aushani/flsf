// adpated from dascar
#pragma once

#include <string>

// OSG
#include <osg/Vec3d>
#include <osg/MatrixTransform>

namespace library {
namespace osg_nodes {

class Car : public osg::MatrixTransform {
 public:
  Car();

 private:
  // file module location (relative to utils.cpp)
  const std::string _k_car_file = "/home/aushani/data/osg_obj/lexus_hs.obj";
  const double _k_scale = 0.075;
  const osg::Vec3d _k_pos = osg::Vec3d(1.5, 0, 0.2);

 protected:
  virtual ~Car() = default;
};

} // namespace osg_nodes
} // namespace library
