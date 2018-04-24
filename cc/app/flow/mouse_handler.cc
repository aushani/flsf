#include "app/flow/mouse_handler.h"

#include <iostream>

#include <Eigen/Geometry>

namespace app {
namespace flow {

MouseHandler::MouseHandler(const std::shared_ptr<App> &app) :
 library::viewer::PickHandler(),
 app_(app) {
}

void MouseHandler::pick(osgViewer::View* view, const osgGA::GUIEventAdapter& ea) {
  osgUtil::LineSegmentIntersector::Intersections intersections;

  bool ctrl = false;
  if (ea.getModKeyMask() && osgGA::GUIEventAdapter::ModKeyMask::MODKEY_CTRL) {
    ctrl = true;
  }

  // Only ctrl+clicks for now
  if (!ctrl) {
    return;
  }

  if (view->computeIntersections(ea, intersections)) {
    for (osgUtil::LineSegmentIntersector::Intersections::iterator hitr = intersections.begin();
         hitr != intersections.end(); ++hitr) {

      osg::Vec3 p = hitr->getWorldIntersectPoint();

      Command command(Type::CLICK_AT);
      command.SetClickPosition(p[0], p[1]);
      app_->QueueCommand(command);

      break;
    }
  }
}

}  // namespace flow
}  // namespace app
