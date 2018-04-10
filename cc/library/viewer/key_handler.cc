#include "library/viewer/key_handler.h"

#include <iostream>

namespace library {
namespace viewer {

bool KeyHandler::handle(const osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter& aa) {
  if (ea.getEventType() == osgGA::GUIEventAdapter::KEYDOWN) {
    KeyPress(ea);
    return false;
  }

  return false;
}

bool KeyHandler::KeyPress(const osgGA::GUIEventAdapter &ea) {
  printf("Got key: %c\n", ea.getKey());
  return true;
}

} // namespace viewer
} // namespace library
