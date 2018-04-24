#pragma once

#include <memory>
#include <queue>

#include <osgGA/GUIActionAdapter>
#include <osgGA/GUIEventAdapter>
#include <osgGA/GUIEventHandler>
#include <osgViewer/CompositeViewer>

#include "library/viewer/key_handler.h"

#include "app/flow/app.h"

namespace app {
namespace flow {

class KeyHandler : public library::viewer::KeyHandler, boost::noncopyable {
 public:
  KeyHandler(const std::shared_ptr<App> &app);

  bool KeyPress(const osgGA::GUIEventAdapter &ea);

 private:
  std::shared_ptr<App> app_;
};

} // namespace flow
} // namespace app
