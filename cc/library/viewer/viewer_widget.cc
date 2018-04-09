// adapted from dascar
#include <iostream>

#include "library/viewer/viewer_widget.h"

namespace library {
namespace viewer {

ViewerWidget::ViewerWidget(QWidget* parent, Qt::WindowFlags f, osgViewer::ViewerBase::ThreadingModel tm)
    : QWidget(parent, f), grid_(new QGridLayout()) {
  setThreadingModel(tm);

  // disable the default setting of viewer.done() by pressing Escape.
  setKeyEventSetsDone(0);

  osg::ref_ptr<osgQt::GraphicsWindowQt> gw = create_graphics_window(0, 0, 1000, 1000);
  QWidget* widget1 = add_view_widget(gw);

  grid_->addWidget(widget1, 0, 0);
  grid_->setContentsMargins(0, 0, 0, 1);  // remove empty space
  setLayout(grid_.get());

  connect(&timer_, SIGNAL(timeout()), this, SLOT(update()));
  timer_.start(10);
}

QWidget* ViewerWidget::add_view_widget(osg::ref_ptr<osgQt::GraphicsWindowQt> gw) {
  view_ = new osgViewer::View();
  addView(view_);

  osg::ref_ptr<osg::Camera> camera = view_->getCamera();
  camera->setGraphicsContext(gw);

  const osg::GraphicsContext::Traits* traits = gw->getTraits();

  camera->setViewport(new osg::Viewport(0, 0, traits->width, traits->height));

  // from osgviewerQt in osg examples
  // camera->setComputeNearFarMode(osg::CullSettings::DO_NOT_COMPUTE_NEAR_FAR);
  camera->setProjectionMatrixAsPerspective(
      30.0f, static_cast<double>(traits->width) / static_cast<double>(traits->height), 1.0f, 10000.0f);

  //_view->setSceneData(scene);
  gw->setTouchEventsEnabled(true);
  return gw->getGLWidget();
}

osg::ref_ptr<osgQt::GraphicsWindowQt> ViewerWidget::create_graphics_window(int x, int y, int w, int h,
                                                                           const std::string& name,
                                                                           bool window_decoration) {
  osg::DisplaySettings* ds = osg::DisplaySettings::instance().get();
  osg::ref_ptr<osg::GraphicsContext::Traits> traits = new osg::GraphicsContext::Traits;

  traits->windowName = name;
  traits->windowDecoration = window_decoration;
  traits->x = x;
  traits->y = y;
  traits->width = w;
  traits->height = h;
  traits->doubleBuffer = true;
  traits->alpha = ds->getMinimumNumAlphaBits();
  traits->stencil = ds->getMinimumNumStencilBits();
  traits->sampleBuffers = ds->getMultiSamples();
  traits->samples = ds->getNumMultiSamples();

  return new osgQt::GraphicsWindowQt(traits.get());
}

void ViewerWidget::frame(double t) {
  lock();
  osgViewer::CompositeViewer::frame();
  unlock();
}

void ViewerWidget::lock() {
  mutex_.lock();
}

void ViewerWidget::unlock() {
  mutex_.unlock();
}

} // namespace viewer
} // namespace library
