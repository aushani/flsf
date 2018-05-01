#pragma once

#include <osgGA/GUIActionAdapter>
#include <osgGA/GUIEventAdapter>
#include <osgGA/GUIEventHandler>
#include <osgViewer/CompositeViewer>

namespace library {
namespace viewer {

class MouseHandler : public osgGA::GUIEventHandler {
 public:
  MouseHandler() {};

  bool handle(const osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter& aa);

  virtual void HandleClick(osgViewer::View* view, const osgGA::GUIEventAdapter& ea) = 0;

 protected:
  osg::Vec3 GetClickLocation(osgViewer::View *view, const osgGA::GUIEventAdapter& ea);
};

} // namespace viewer
} // namespace library
