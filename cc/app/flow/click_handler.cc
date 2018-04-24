#include "app/flow/click_handler.h"

#include <iostream>

#include <Eigen/Geometry>

namespace app {
namespace flow {

ClickHandler::ClickHandler(const std::shared_ptr<App> &app) :
 library::viewer::PickHandler(),
 app_(app) {
}

void ClickHandler::pick(osgViewer::View* view, const osgGA::GUIEventAdapter& ea) {
  osgUtil::LineSegmentIntersector::Intersections intersections;

  if (view->computeIntersections(ea, intersections)) {
    for (osgUtil::LineSegmentIntersector::Intersections::iterator hitr = intersections.begin();
         hitr != intersections.end(); ++hitr) {

      osg::Vec3 p = hitr->getWorldIntersectPoint();

      printf("clock at %f %f\n", p[0], p[1]);
      break;
    }
  }
}

}  // namespace flow
}  // namespace app
