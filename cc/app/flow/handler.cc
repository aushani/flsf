#include "app/flow/handler.h"

namespace app {
namespace flow {

Handler::Handler() :
 library::viewer::KeyHandler() {
}

bool Handler::KeyPress(const osgGA::GUIEventAdapter& ea) {
  printf("Got key: %c\n", ea.getKey());
  return true;
}

void Handler::SetApp(const std::shared_ptr<App> &app) {
  app_ = app;
}

}  // namespace flow
}  // namespace app
