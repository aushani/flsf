#include "app/flow/key_handler.h"

namespace app {
namespace flow {

KeyHandler::KeyHandler(const std::shared_ptr<App> &app) :
 library::viewer::KeyHandler(),
 app_(app) {
}

bool KeyHandler::KeyPress(const osgGA::GUIEventAdapter& ea) {
  int key = ea.getKey();
  bool ctrl = ea.getModKeyMask() && osgGA::GUIEventAdapter::ModKeyMask::MODKEY_CTRL;

  //printf("Got key: %c (0x%X, 0x%X) with mask 0x%X\n", key, ea.getKey(), ea.getUnmodifiedKey(), ea.getModKeyMask());

  if (!ctrl) {
    if (key == osgGA::GUIEventAdapter::KeySymbol::KEY_C) {
      Command command(Type::CLEAR_DM);
      app_->QueueCommand(command);
      return true;
    }

    if (key == osgGA::GUIEventAdapter::KeySymbol::KEY_N) {
      Command command(Type::NEXT);
      app_->QueueCommand(command);
      return true;
    }

    if (key == osgGA::GUIEventAdapter::KeySymbol::KEY_1) {
      int view_mode = 1;

      Command command(Type::VIEW_MODE);
      command.SetViewMode(view_mode);

      app_->QueueCommand(command);
      return true;
    }

    if (key == osgGA::GUIEventAdapter::KeySymbol::KEY_2) {
      int view_mode = 2;

      Command command(Type::VIEW_MODE);
      command.SetViewMode(view_mode);

      app_->QueueCommand(command);
      return true;
    }

    if (key == osgGA::GUIEventAdapter::KeySymbol::KEY_3) {
      int view_mode = 3;

      Command command(Type::VIEW_MODE);
      command.SetViewMode(view_mode);

      app_->QueueCommand(command);
      return true;
    }

    if (key == osgGA::GUIEventAdapter::KeySymbol::KEY_4) {
      int view_mode = 4;

      Command command(Type::VIEW_MODE);
      command.SetViewMode(view_mode);

      app_->QueueCommand(command);
      return true;
    }

    if (key == osgGA::GUIEventAdapter::KeySymbol::KEY_5) {
      int view_mode = 5;

      Command command(Type::VIEW_MODE);
      command.SetViewMode(view_mode);

      app_->QueueCommand(command);
      return true;
    }
  }

  return false;
}

}  // namespace flow
}  // namespace app
