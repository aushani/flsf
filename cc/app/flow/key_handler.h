#pragma once

#include <memory>
#include <queue>
#include <thread>

#include <osgGA/GUIActionAdapter>
#include <osgGA/GUIEventAdapter>
#include <osgGA/GUIEventHandler>
#include <osgViewer/CompositeViewer>

#include "library/viewer/key_handler.h"

#include "app/flow/app.h"
#include "app/flow/command_queue.h"

namespace app {
namespace flow {

class KeyHandler : public library::viewer::KeyHandler, boost::noncopyable {
 public:
  KeyHandler(const std::shared_ptr<App> &app);
  ~KeyHandler();

  KeyHandler(const KeyHandler &h);
  KeyHandler& operator=(const KeyHandler &h);

  bool KeyPress(const osgGA::GUIEventAdapter &ea);

 private:
  std::shared_ptr<App> app_;
  std::shared_ptr<CommandQueue> command_queue_;

  std::thread processing_thread_;

  bool running_ = true;

  void Run();
};

} // namespace flow
} // namespace app
