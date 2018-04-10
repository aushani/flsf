#include "app/flow/handler.h"

namespace app {
namespace flow {

Handler::Handler() :
 library::viewer::PickHandler() {
}

void Handler::pick(osgViewer::View* view, const osgGA::GUIEventAdapter& ea) {
  osgUtil::LineSegmentIntersector::Intersections intersections;

  printf("Hello\n");

  if (view->computeIntersections(ea, intersections)) {
    for (auto hitr = intersections.begin(); hitr != intersections.end(); ++hitr) {
    }
  }
}

void Handler::SetApp(const std::shared_ptr<App> &app) {
  app_ = app;
}

}  // namespace flow
}  // namespace app
