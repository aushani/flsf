// adapted from dascar
#pragma once

#include <memory>

#include <QtGui/QProgressBar>
#include <QtGui/QApplication>
#include <QtGui/QSpinBox>
#include <QtGui/QDoubleSpinBox>
#include <QtGui/QCheckBox>
#include <osg/MatrixTransform>
#include <osgGA/GUIEventHandler>
#include <osgQt/GraphicsWindowQt>
#include <osgViewer/CompositeViewer>

#include <QTimer>
#include <QGridLayout>

#include "library/viewer/viewer_window.h"

namespace library {
namespace viewer {

class Viewer {
 public:
  Viewer(osg::ArgumentParser *args);

  void AddChild(osg::Node *n);
  void RemoveAllChildren();

  void AddHandler(osgGA::GUIEventHandler *h);

  void Start();

  void Lock();
  void Unlock();

 private:
  std::unique_ptr<QApplication> qapp_;
  std::unique_ptr<ViewerWindow> vwindow_;
};

} // namespace viewer
} // namespace library
