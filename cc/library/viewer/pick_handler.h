#pragma once

#include <osgGA/GUIActionAdapter>
#include <osgGA/GUIEventAdapter>
#include <osgGA/GUIEventHandler>
#include <osgViewer/CompositeViewer>

namespace library {
namespace viewer {

// from osgpick example
// class to handle events with a pick
class PickHandler : public osgGA::GUIEventHandler {
 public:
  //PickHandler(State* state) : _state(state){};
  PickHandler() {};

  bool handle(const osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter& aa);

  virtual void pick(osgViewer::View* view, const osgGA::GUIEventAdapter& ea);

 private:
  //State* _state;
};

} // namespace viewer
} // namespace library
