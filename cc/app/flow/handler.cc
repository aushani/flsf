#include "app/flow/handler.h"

namespace app {
namespace flow {

Handler::Handler() :
 library::viewer::KeyHandler() {
}

bool Handler::KeyPress(const osgGA::GUIEventAdapter& ea) {
  bool ctrl = ea.getModKeyMask() && osgGA::GUIEventAdapter::ModKeyMask::MODKEY_CTRL;

  if (ctrl) {
    char c = ea.getKey() + 'A' - 1;
    return KeyCommand(c);
  }

  return false;
}

void Handler::SetApp(const std::shared_ptr<App> &app) {
  app_ = app;
}

bool Handler::KeyCommand(char c) {
  switch(c) {
    case 'N':
      app_->ProcessNext();
      return true;

    default:
      printf("Invalid command: %c\n", c);
      return false;
  }
}

}  // namespace flow
}  // namespace app
