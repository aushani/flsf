#include "app/flow/handler.h"

namespace app {
namespace flow {

Handler::Handler(const std::shared_ptr<App> &app) :
 library::viewer::KeyHandler(),
 app_(app),
 command_queue_(std::make_shared<CommandQueue>()),
 processing_thread_(&Handler::Run, *this) {
}

Handler::~Handler() {
  running_ = false;
  processing_thread_.join();
}

Handler::Handler(const Handler &h) :
 library::viewer::KeyHandler(),
 app_(h.app_),
 command_queue_(h.command_queue_),
 processing_thread_(&Handler::Run, this) {
}

Handler& Handler::operator=(const Handler &h) {
  app_ = h.app_;
  command_queue_ = h.command_queue_;
  return *this;
}

bool Handler::KeyPress(const osgGA::GUIEventAdapter& ea) {
  bool ctrl = ea.getModKeyMask() && osgGA::GUIEventAdapter::ModKeyMask::MODKEY_CTRL;

  if (ctrl) {
    char c = ea.getKey() + 'A' - 1;
    return command_queue_->Push(c);
  }

  return false;
}

void Handler::Run() {

  while (running_) {
    Command c = command_queue_->Pop(1000);

    switch(c) {
      case NEXT:
        app_->ProcessNext();
        break;

      case NONE:
        break;
    }
  }
}

}  // namespace flow
}  // namespace app
