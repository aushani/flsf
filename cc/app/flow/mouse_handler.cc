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

  osg::Vec3 world_pos = GetClickLocation(view, ea);

  // Now send command
  Command command(Type::CLICK_AT);
  command.SetClickPosition(world_pos[0], world_pos[1]);
  app_->QueueCommand(command);
}

}  // namespace flow
}  // namespace app
