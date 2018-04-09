// adapted from dascar
#include "library/viewer/viewer.h"

#include <iostream>
#include <QLabel>

namespace library {
namespace viewer {

Viewer::Viewer(osg::ArgumentParser *args) :
  qapp_(new QApplication(args->argc(), args->argv())),
  vwindow_(new ViewerWindow(args, 0, Qt::Widget)) {
}

void Viewer::AddChild(osg::Node *n) {
  vwindow_->AddChild(n);
}

void Viewer::AddHandler(osgGA::GUIEventHandler *h) {
  vwindow_->AddHandler(h);
}

void Viewer::RemoveAllChildren() {
  vwindow_->RemoveAllChildren();
}

void Viewer::Start() {
  int rc = vwindow_->Start();

  if (rc != EXIT_SUCCESS) {
    return;
  }
  qapp_->exec();
}

void Viewer::Lock() {
  vwindow_->Lock();
}

void Viewer::Unlock() {
  vwindow_->Unlock();
}

}  // namespace viewer
}  // namespace library
