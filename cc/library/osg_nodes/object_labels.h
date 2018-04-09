#pragma once

#include <osg/MatrixTransform>
#include <osg/Vec4>

#include "library/kitti/object_label.h"

namespace library {
namespace osg_nodes {

class ObjectLabels : public osg::Group {
 public:
  ObjectLabels(const library::kitti::ObjectLabels &labels, const Eigen::Matrix4d &T_cv, bool gt=true);
 private:
  osg::Vec4 color_car_        = osg::Vec4(1.0, 0.0, 0.0, 0.8);
  osg::Vec4 color_cyclist_    = osg::Vec4(0.0, 1.0, 0.0, 0.8);
  osg::Vec4 color_pedestrian_ = osg::Vec4(0.0, 0.0, 1.0, 0.8);

  osg::Vec4 color_other_      = osg::Vec4(0.8, 0.8, 0.8, 0.8);
};

}  // namespace osg_nodes
}  // namespace library
