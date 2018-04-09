#include "library/osg_nodes/object_labels.h"

#include <iostream>

#include <Eigen/Dense>

#include <osg/PolygonMode>
#include <osg/LineWidth>
#include <osg/ShapeDrawable>

#include "library/osg_nodes/colorful_box.h"
#include "library/osg_nodes/composite_shape_group.h"

namespace kt = library::kitti;

namespace library {
namespace osg_nodes {

ObjectLabels::ObjectLabels(const kt::ObjectLabels &labels, const Eigen::Matrix4d &T_cv, bool gt) : osg::Group() {
  osg::ref_ptr<CompositeShapeGroup> csg = new CompositeShapeGroup();

  osg::ref_ptr<osg::StateSet> ss = csg->getOrCreateStateSet();
  osg::ref_ptr<osg::PolygonMode> pm = new osg::PolygonMode(osg::PolygonMode::FRONT_AND_BACK, osg::PolygonMode::LINE);
  osg::ref_ptr<osg::LineWidth> lw = new osg::LineWidth(8.0);
  ss->setAttributeAndModes(pm, osg::StateAttribute::ON | osg::StateAttribute::OVERRIDE);
  ss->setAttributeAndModes(lw, osg::StateAttribute::ON | osg::StateAttribute::OVERRIDE);

  for (const auto &label : labels) {
    Eigen::Vector4d p_camera(label.location[0], label.location[1], label.location[2], 1);
    auto p_vel = T_cv.inverse() * p_camera;
    osg::Vec3 pos(p_vel.x(), p_vel.y(), p_vel.z() + 0.8); // offset ?

    osg::Quat quat(-label.rotation_y, osg::Vec3(0, 0, 1));  // rotation only in heading

    osg::ref_ptr<osg::Box> box = new osg::Box(pos, label.dimensions[1], label.dimensions[2], label.dimensions[0]);
    box->setRotation(quat);

    osg::ref_ptr<osg::ShapeDrawable> shape = new osg::ShapeDrawable(box);
    shape->setColor(osg::Vec4(0.9, 0.9, 0.9, 1.0));

    if (gt) {
      if (label.type == kt::ObjectLabel::CAR) {
        shape->setColor(color_car_);
      } else if (label.type == kt::ObjectLabel::PEDESTRIAN) {
        shape->setColor(color_pedestrian_);
      } else if (label.type == kt::ObjectLabel::CYCLIST) {
        shape->setColor(color_cyclist_);
      } else {
        shape->setColor(color_other_);
      }
    }

    csg->addChild(shape);
  }

  addChild(csg);
}

}  // namespace osg_nodes
}  // namespace library
