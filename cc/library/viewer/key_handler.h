#pragma once

#include <osgGA/GUIActionAdapter>
#include <osgGA/GUIEventAdapter>
#include <osgGA/GUIEventHandler>
#include <osgViewer/CompositeViewer>

namespace library {
namespace viewer {

class KeyHandler : public osgGA::GUIEventHandler {
 public:
  KeyHandler() {};

  bool handle(const osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter& aa);

  virtual bool KeyPress(const osgGA::GUIEventAdapter &ea);

};

} // namespace viewer
} // namespace library
