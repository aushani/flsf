// Adapted from dascar
#include "library/kitti/nodes/tracklets.h"

#include <iostream>

#include <osg/PolygonMode>
#include <osg/LineWidth>
#include <osg/ShapeDrawable>

//#include "library/osg_nodes/colorful_box.h"
#include "library/osg_nodes/composite_shape_group.h"

namespace osgn = library::osg_nodes;

namespace library {
namespace kitti {
namespace nodes {

Tracklets::Tracklets() :
 osg::Group() {
}

Tracklets::Tracklets(kt::Tracklets *tracklets, int frame) :
 osg::Group() {
  Update(tracklets, frame);
}

void Tracklets::Update(kt::Tracklets *tracklets, int frame) {
  // Clear any old tracklets
  while (getNumChildren() > 0) {
    removeChild(0, 1);
  }

  // Render active tracks
  int n = tracklets->numberOfTracklets();

  kt::Tracklets::tPose *pose;
  kt::Tracklets::tTracklet *t;

  osg::ref_ptr<osgn::CompositeShapeGroup> csg = new osgn::CompositeShapeGroup();

  osg::ref_ptr<osg::StateSet> ss = csg->getOrCreateStateSet();
  osg::ref_ptr<osg::PolygonMode> pm = new osg::PolygonMode(osg::PolygonMode::FRONT_AND_BACK, osg::PolygonMode::LINE);
  osg::ref_ptr<osg::LineWidth> lw = new osg::LineWidth(8.0);
  ss->setAttributeAndModes(pm, osg::StateAttribute::ON | osg::StateAttribute::OVERRIDE);
  ss->setAttributeAndModes(lw, osg::StateAttribute::ON | osg::StateAttribute::OVERRIDE);

  for (int i = 0; i < n; i++) {
    if (tracklets->getPose(i, frame, pose)) {
      t = tracklets->getTracklet(i);

      osg::Vec3 pos(pose->tx, pose->ty, pose->tz + 0.8);  // offset?
      osg::Quat quat(pose->rz, osg::Vec3(0, 0, 1));  // rotation only in heading

      osg::ref_ptr<osg::Box> box = new osg::Box(pos, t->l, t->w, t->h);
      box->setRotation(quat);

      osg::ref_ptr<osg::ShapeDrawable> shape = new osg::ShapeDrawable(box);

      std::string type = t->objectType;
      // std::cout << "type: " << type << std::endl;

      if (type == "Car") {
        shape->setColor(color_car_);
      } else if (type == "Pedestrian" || type == "Person") {
        shape->setColor(color_pedestrian_);
      } else if (type == "Cyclist") {
        shape->setColor(color_cyclist_);
      } else {
        shape->setColor(color_other_);
      }

      csg->addChild(shape);

      // printf("Track %d is active at frame %d\n", i, n);
    }
  }

  addChild(csg);
}

void Tracklets::Render(bool render) {
  setNodeMask(render);
}

}  // namespace nodes
}  // namespace kitti
}  // namespace library
