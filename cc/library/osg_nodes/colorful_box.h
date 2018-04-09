// Adapted from dascar
#pragma once

#include <osg/ShapeDrawable>

namespace library {
namespace osg_nodes {

class ColorfulBox : public osg::ShapeDrawable {
 public:
  ColorfulBox(osg::Vec4 color, osg::Vec3 pos, double scale);

 protected:
  virtual ~ColorfulBox() = default;
};

}  // namespace osg_nodes
}  // namespace library
