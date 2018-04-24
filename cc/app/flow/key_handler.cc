#include "app/flow/key_handler.h"

namespace app {
namespace flow {

KeyHandler::KeyHandler(const std::shared_ptr<App> &app) :
 library::viewer::KeyHandler(),
 app_(app) {
}

bool KeyHandler::KeyPress(const osgGA::GUIEventAdapter& ea) {
  bool ctrl = ea.getModKeyMask() && osgGA::GUIEventAdapter::ModKeyMask::MODKEY_CTRL;

  if (ctrl) {
    char c = ea.getKey() + 'A' - 1;

    switch(c) {
      case 'N':
        Command command(Type::NEXT);
        app_->QueueCommand(command);
        return true;
    }
  }

  return false;
}

}  // namespace flow
}  // namespace app
