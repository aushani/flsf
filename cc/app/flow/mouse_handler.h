#pragma once

#include <memory>

#include <osgGA/GUIActionAdapter>
#include <osgGA/GUIEventAdapter>
#include <osgGA/GUIEventHandler>
#include <osgViewer/CompositeViewer>

#include "library/viewer/mouse_handler.h"

#include "app/flow/app.h"

namespace app {
namespace flow {

class MouseHandler : public library::viewer::MouseHandler {
 public:
  MouseHandler(const std::shared_ptr<App> &app);

  void HandleClick(osgViewer::View* view, const osgGA::GUIEventAdapter& ea);

 private:
  std::shared_ptr<App> app_;
};

} // namespace flow
} // namespace app
