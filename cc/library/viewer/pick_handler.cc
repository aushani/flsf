// from osgpick example
//
#include "library/viewer/pick_handler.h"

namespace library {
namespace viewer {

bool PickHandler::handle(const osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter& aa) {
  switch (ea.getEventType()) {
    case (osgGA::GUIEventAdapter::PUSH): {
      osgViewer::View* view = dynamic_cast<osgViewer::View*>(&aa);
      if (view) {
        pick(view, ea);
      }
      return false;
    }
    default: {
      return false;
    }
  }
}

void PickHandler::pick(osgViewer::View* view, const osgGA::GUIEventAdapter& ea) {
  int mod = ea.getModKeyMask();

  bool ctrl = false;
  if (mod && osgGA::GUIEventAdapter::ModKeyMask::MODKEY_CTRL) ctrl = true;

  osgUtil::LineSegmentIntersector::Intersections intersections;

  if (view->computeIntersections(ea, intersections)) {
    for (osgUtil::LineSegmentIntersector::Intersections::iterator hitr = intersections.begin();
         hitr != intersections.end(); ++hitr) {
      osg::Vec3 p = hitr->getWorldIntersectPoint();

      // is it valid?
      if (!hitr->drawable.valid()) { continue; }

      // is it a Tetrahedron?
      std::string drawable_name(hitr->drawable->className());
      printf("Drawable name: %s\n", drawable_name.c_str());

      // does it have a parent?
      osg::Group* group = hitr->drawable->getParent(0);
      if (!group) { printf("no ground\n"); continue; }

      // is that parent an occ grid?
      // std::string parent_name = group->className();
      // printf("parent name: %s\n", parent_name.c_str());
      // if (parent_name != osg::FactorGraph::Node::get_class_name())
      //{ continue; }

      // cast to node
      // osg::FactorGraph::Node* n = (osg::FactorGraph::Node*) group;

      // if (!n->get_point_cloud_shown()) {
      //    int64_t utime = n->get_node()->get_node_utime();
      //    velodyne_returns_t* vr = _state->fa->get_point_cloud(utime);
      //    n->show_point_cloud(vr);
      //    velodyne_returns_destroy(vr);
      //} else {
      //    n->hide_point_cloud();
      //}

      break;  // only do first one
    }
  }
}

} // namespace viewer
} // namespace library
