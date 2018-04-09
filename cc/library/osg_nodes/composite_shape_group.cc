// Adapted from dascar
#include "library/osg_nodes/composite_shape_group.h"

#include <osg/BlendFunc>

namespace library {
namespace osg_nodes {

CompositeShapeGroup::CompositeShapeGroup()
    : osg::Group(), cshape_(new osg::CompositeShape), sdrawable_(new osg::ShapeDrawable(cshape_)) {
  // use alpha to draw transparent objects
  osg::ref_ptr<osg::StateSet> set = sdrawable_->getOrCreateStateSet();
  set->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
  set->setAttributeAndModes(new osg::BlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA), osg::StateAttribute::ON);

  addChild(sdrawable_);
}

}  // namespace osg_nodes
}  // namespace library
