#include "app/viewer/simple_handler.h"

#include <iostream>

#include <Eigen/Geometry>

namespace app {
namespace viewer {

SimpleHandler::SimpleHandler(const kt::Tracklets &tracklets, int frame) :
  library::viewer::PickHandler(),
  tracklets_(tracklets),
  frame_(frame) {

}

void SimpleHandler::pick(osgViewer::View* view, const osgGA::GUIEventAdapter& ea) {
  osgUtil::LineSegmentIntersector::Intersections intersections;

  if (view->computeIntersections(ea, intersections)) {
    for (osgUtil::LineSegmentIntersector::Intersections::iterator hitr = intersections.begin();
         hitr != intersections.end(); ++hitr) {

      osg::Vec3 p = hitr->getWorldIntersectPoint();

      auto label = GetClass(p[0], p[1]);
      printf("Class: %s\n", label.c_str());

      break;
    }
  }
}

std::string SimpleHandler::GetClass(double x, double y) {
  kt::Tracklets::tPose* pose = nullptr;

  for (int t_id=0; t_id<tracklets_.numberOfTracklets(); t_id++) {
    if (!tracklets_.isActive(t_id, frame_)) {
      continue;
    }

    tracklets_.getPose(t_id, frame_, pose);
    auto tt = tracklets_.getTracklet(t_id);

    Eigen::Affine3d rx(Eigen::AngleAxisd(pose->rx, Eigen::Vector3d(1, 0, 0)));
    Eigen::Affine3d ry(Eigen::AngleAxisd(pose->ry, Eigen::Vector3d(0, 1, 0)));
    Eigen::Affine3d rz(Eigen::AngleAxisd(pose->rz, Eigen::Vector3d(0, 0, 1)));
    auto r = rx * ry * rz;
    Eigen::Affine3d t(Eigen::Translation3d(Eigen::Vector3d(pose->tx, pose->ty, pose->tz)));
    Eigen::Matrix4d X_tw = (t*r).matrix();
    Eigen::Matrix4d X_wt = X_tw.inverse();
    Eigen::Vector3d x_w(x, y, 0);
    Eigen::Vector4d x_th = (X_wt * x_w.homogeneous());
    Eigen::Vector3d x_t = x_th.hnormalized();

    // Check if we're inside this track, otherwise this is not the track we
    // are looking for...
    if (std::fabs(x_t.x())<tt->l/2 && std::fabs(x_t.y())<tt->w/2) {
        return tt->objectType;
    }
  }

  // This is background
  return "Background";
}

}  // namespace viewer
}  // namespace app
