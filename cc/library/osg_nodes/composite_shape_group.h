// Adapted from dascar
#pragma once

#include <osg/Group>
#include <osg/ShapeDrawable>
#include <osg/Shape>

namespace library {
namespace osg_nodes {

class CompositeShapeGroup : public osg::Group {
 public:
  CompositeShapeGroup();

  osg::ref_ptr<osg::CompositeShape> GetCShape() const { return cshape_; }

  osg::ref_ptr<osg::ShapeDrawable> GetSDrawable() const { return sdrawable_; }

 private:
  osg::ref_ptr<osg::CompositeShape> cshape_;
  osg::ref_ptr<osg::ShapeDrawable> sdrawable_;

 protected:
  virtual ~CompositeShapeGroup() = default;
};

}  // namespace osg_nodes
}  // namespace library
