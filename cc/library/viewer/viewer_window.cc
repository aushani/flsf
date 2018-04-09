// adapted from dascar
#include "library/viewer/viewer_window.h"

#include <iostream>

#include <osgGA/NodeTrackerManipulator>
#include <osgGA/KeySwitchMatrixManipulator>
#include <osgGA/StateSetManipulator>
#include <osgGA/TerrainManipulator>
#include <osgViewer/ViewerEventHandlers>

#include "library/util/angle.h"

#include "library/viewer/pick_handler.h"
#include "library/viewer/terrain_trackpad_manipulator.h"

namespace ut = library::util;

namespace library {
namespace viewer {

ViewerWindow::ViewerWindow(osg::ArgumentParser *args, QWidget *parent, Qt::WindowFlags f)
    : QMainWindow(parent, f), xform_car_(new osg::MatrixTransform()) {
  osgViewer::ViewerBase::ThreadingModel tm = osgViewer::ViewerBase::SingleThreaded;
  vwidget_ = new ViewerWidget(0, Qt::Widget, tm);

  setCentralWidget(vwidget_);

  setWindowTitle(tr("Viewer"));
  setMinimumSize(640, 480);

  Init(args);
}

void ViewerWindow::Init(osg::ArgumentParser *args) {
  osg::ApplicationUsage *au = args->getApplicationUsage();

  osg::ref_ptr<osgViewer::View> view = vwidget_->GetView();

  // TODO: magic numbers
  view->getCamera()->setClearColor(osg::Vec4d(1, 1, 1, 0)); // white
  //view->getCamera()->setClearColor(osg::Vec4d(0, 0, 0, 0)); // black

  osg::ref_ptr<osgGA::KeySwitchMatrixManipulator> ksm = new osgGA::KeySwitchMatrixManipulator();

  ksm->addMatrixManipulator('1', "TerrainTrackpad", new TerrainTrackpadManipulator());
  ksm->addMatrixManipulator('2', "NodeTracker", new osgGA::NodeTrackerManipulator());
  ksm->addMatrixManipulator('3', "Terrain", new osgGA::TerrainManipulator());

  // set initial camera position (for all manipulators)
  // TODO: magic numbers
  ksm->setHomePosition(osg::Vec3d(0, 0, 100), osg::Vec3d(0, 0, 0), osg::Vec3d(1, 0, 0), false);

  ksm->getUsage(*au);
  view->setCameraManipulator(ksm.get());

  // add the state manipulator
  osg::ref_ptr<osgGA::StateSetManipulator> ssm =
      new osgGA::StateSetManipulator(view->getCamera()->getOrCreateStateSet());
  ssm->getUsage(*au);
  view->addEventHandler(ssm);

  // add the stats handler
  osg::ref_ptr<osgViewer::StatsHandler> sh = new osgViewer::StatsHandler();
  sh->getUsage(*au);
  view->addEventHandler(sh);

  // add the help handler
  osg::ref_ptr<osgViewer::HelpHandler> hh = new osgViewer::HelpHandler(au);
  hh->getUsage(*au);
  view->addEventHandler(hh);

  // add the screen capture handler
  osg::ref_ptr<osgViewer::ScreenCaptureHandler> sch = new osgViewer::ScreenCaptureHandler();
  sch->getUsage(*au);
  view->addEventHandler(sch);

  // add the level of detail scale selector
  osg::ref_ptr<osgViewer::LODScaleHandler> lod = new osgViewer::LODScaleHandler();
  lod->getUsage(*au);
  view->addEventHandler(lod);

  // rotate by x until z down
  // car RH coordinate frame has x forward, z down
  osg::Matrixd H(osg::Quat(ut::DegreesToRadians(180), osg::Vec3d(1, 0, 0)));
  // osg::Matrixd H(osg::Quat(0, osg::Vec3d(1, 0, 0)));
  osg::ref_ptr<osg::MatrixTransform> xform = new osg::MatrixTransform(H);

  osg::Matrixd D(osg::Quat(M_PI, osg::Vec3d(1, 0, 0)));
  D.postMultTranslate(osg::Vec3d(0, 0, 0));
  xform_car_->setMatrix(D);
  // xform_car->addChild(new osg::Axes());
  xform->addChild(xform_car_);

  // set scene
  view->setSceneData(xform);
}

void ViewerWindow::AddChild(osg::Node *n) {
  vwidget_->lock();

  auto ss = n->getOrCreateStateSet();
  //ss->setMode(GL_BLEND, osg::StateAttribute::ON);
  //ss->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
  ss->setRenderBinDetails(bin_at_++, "RenderBin");

  xform_car_->addChild(n);

  vwidget_->unlock();
}

void ViewerWindow::RemoveAllChildren() {
  vwidget_->lock();
  while (xform_car_->getNumChildren() > 0) {
    xform_car_->removeChild(0, 1);
  }
  vwidget_->unlock();
}

void ViewerWindow::AddHandler(osgGA::GUIEventHandler *h) {
  osg::ref_ptr<osgViewer::View> view = vwidget_->GetView();
  view->addEventHandler(h);
}

int ViewerWindow::Start() {
  show();

  // start threads
  std::cout << "Starting thread..." << std::endl;
  run_thread_ = std::thread(&ViewerWindow::RunThread, this);

  return EXIT_SUCCESS;
}

void ViewerWindow::SlotCleanup() {
  printf("TODO: SlotCleanup\n");
}

void ViewerWindow::RunThread() {
  while (true) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

void ViewerWindow::Lock() {
  vwidget_->lock();
}

void ViewerWindow::Unlock() {
  vwidget_->unlock();
}

}  // namespace viewer
}  // namespace library
