#pragma once

#include <memory>

#include <osgGA/GUIActionAdapter>
#include <osgGA/GUIEventAdapter>
#include <osgGA/GUIEventHandler>
#include <osgViewer/CompositeViewer>

#include "library/viewer/key_handler.h"

#include "app/flow/app.h"

namespace app {
namespace flow {

// from osgpick example
// class to handle events with a pick
class Handler : public library::viewer::KeyHandler {
 public:
  Handler();

  bool KeyPress(const osgGA::GUIEventAdapter& ea);

  void SetApp(const std::shared_ptr<App> &app);

 private:
  std::shared_ptr<App> app_;
};

} // namespace flow
} // namespace app
