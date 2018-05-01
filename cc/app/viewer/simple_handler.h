#pragma once

#include <osgGA/GUIActionAdapter>
#include <osgGA/GUIEventAdapter>
#include <osgGA/GUIEventHandler>
#include <osgViewer/CompositeViewer>

#include "library/kitti/tracklets.h"
#include "library/kitti/camera_cal.h"
#include "library/viewer/mouse_handler.h"

namespace kt = library::kitti;
namespace fs = boost::filesystem;

namespace app {
namespace viewer {

class SimpleHandler : public library::viewer::MouseHandler {
 public:
  SimpleHandler(const fs::path &base_path);

  void SetFrame(int frame);

  void HandleClick(osgViewer::View* view, const osgGA::GUIEventAdapter& ea);

 private:
  kt::Tracklets tracklets_;
  int frame_ = 0;

  kt::CameraCal camera_cal_;

  std::string GetClass(double x, double y);
};

} // namespace viewer
} // namespace app
