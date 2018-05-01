#include "library/viewer/mouse_handler.h"

namespace library {
namespace viewer {

bool MouseHandler::handle(const osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter& aa) {
  switch (ea.getEventType()) {
    case (osgGA::GUIEventAdapter::PUSH): {
      osgViewer::View* view = dynamic_cast<osgViewer::View*>(&aa);
      if (view) {
        HandleClick(view, ea);
      }
      return false;
    }
    default: {
      return false;
    }
  }
}

osg::Vec3 MouseHandler::GetClickLocation(osgViewer::View *view, const osgGA::GUIEventAdapter& ea) {
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

  return worldPosition;
}

} // namespace viewer
} // namespace library
