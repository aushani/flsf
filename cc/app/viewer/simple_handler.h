#pragma once

#include <osgGA/GUIActionAdapter>
#include <osgGA/GUIEventAdapter>
#include <osgGA/GUIEventHandler>
#include <osgViewer/CompositeViewer>

#include "library/kitti/tracklets.h"
#include "library/viewer/pick_handler.h"

namespace kt = library::kitti;

namespace app {
namespace viewer {

// from osgpick example
// class to handle events with a pick
class SimpleHandler : public library::viewer::PickHandler {
 public:
  SimpleHandler(const kt::Tracklets &tracklets, int frame);

  void pick(osgViewer::View* view, const osgGA::GUIEventAdapter& ea);

 private:
  kt::Tracklets tracklets_;
  int frame_ = 0;

  std::string GetClass(double x, double y);
};

} // namespace viewer
} // namespace app
