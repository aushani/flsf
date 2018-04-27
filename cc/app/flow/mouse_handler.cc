#include "app/flow/mouse_handler.h"

#include <iostream>

#include <Eigen/Geometry>

namespace app {
namespace flow {

MouseHandler::MouseHandler(const std::shared_ptr<App> &app) :
 library::viewer::MouseHandler(),
 app_(app) {
}

void MouseHandler::HandleClick(osgViewer::View* view, const osgGA::GUIEventAdapter& ea) {
  bool ctrl = false;
  if (ea.getModKeyMask() && osgGA::GUIEventAdapter::ModKeyMask::MODKEY_CTRL) {
    ctrl = true;
  }

  // Only ctrl+clicks for now
  if (!ctrl) {
    return;
  }

  // Get camera params
  osg::Camera *camera = view->getCamera();
  osg::Matrixd viewMatrix = camera->getViewMatrix();

  // Find click at ground plane
  osg::Matrix projectionMatrix = camera->getProjectionMatrix();
  osg::Matrix inverseCameraMatrix;
  inverseCameraMatrix.invert(viewMatrix * projectionMatrix );
  osg::Vec3d eye,centre,up;
  camera->getViewMatrixAsLookAt(eye,centre,up);

  // Do some math
  double x_mouse = ea.getXnormalized();
  double y_mouse = ea.getYnormalized();

  double a = inverseCameraMatrix(0, 2);
  double b = inverseCameraMatrix(1, 2);
  double c = inverseCameraMatrix(2, 2);
  double d = inverseCameraMatrix(3, 2);

  double z = -(x_mouse * a + y_mouse * b + d) / c;

  osg::Vec3 click(x_mouse, y_mouse, z);
  osg::Vec3 worldPosition = click * inverseCameraMatrix;

  // Now send command
  Command command(Type::CLICK_AT);
  command.SetClickPosition(worldPosition[0], worldPosition[1]);
  app_->QueueCommand(command);
}

}  // namespace flow
}  // namespace app
