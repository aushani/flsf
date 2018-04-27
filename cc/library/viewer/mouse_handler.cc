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

} // namespace viewer
} // namespace library
