// adpated from dascar
#pragma once

#include <string>

#include <boost/filesystem.hpp>
#include <osg/Vec3d>
#include <osg/MatrixTransform>

namespace fs = boost::filesystem;

namespace library {
namespace osg_nodes {

class Car : public osg::MatrixTransform {
 public:
  Car(const fs::path& obj_file);

 private:
  const double _k_scale = 0.075;
  const osg::Vec3d _k_pos = osg::Vec3d(0.2, 0, -1.65);

 protected:
  virtual ~Car() = default;
};

} // namespace osg_nodes
} // namespace library
