#include "library/kitti/util.h"

namespace library {
namespace kitti {

Eigen::Vector2f FindCorrespondingPosition(Tracklets *tracklets,
                                          const Eigen::Vector2f &pos,
                                          int scan_at,
                                          int scan_des,
                                          const Pose &p1,
                                          const Pose &p2,
                                          float res) {
  pm::Pose3d::Vector6Type x_1;
  x_1 << pos.x(), pos.y(), 0, 0, 0, 0;
  pm::Pose3d x_1_p(x_1);

  // Check if selected position is in any tracklets
  Tracklets::tPose *tp;
  Tracklets::tTracklet *tt;
  for (int i=0; i<tracklets->numberOfTracklets(); i++) {
    if (!tracklets->getPose(i, scan_at, tp)) {
        continue;
    }

    tt = tracklets->getTracklet(i);

    // Is this point within tracklet?
    pm::Pose3d::Vector6Type t_1;
    t_1 << tp->tx, tp->ty, tp->tz, tp->rx, tp->ry, tp->rz;
    pm::Pose3d t_1_p(t_1);

    pm::Pose3d x_t_p = t_1_p.TailToTail(x_1_p);
    pm::Pose3d::Vector6Type x_t = x_t_p.xyzrph();

    // Check if we're inside this track, otherwise this is not the track we
    // are looking for...
    if (fabs(x_t[0])<(tt->l/2 + res) && fabs(x_t[1])<(tt->w/2 + res)) {
        //printf("Inside!\n");
    } else {
        continue;
    }

    // Now project to next frame
    if (!tracklets->getPose(i, scan_des, tp)) {
        continue;
    }

    pm::Pose3d::Vector6Type t_2;
    t_2 << tp->tx, tp->ty, tp->tz, tp->rx, tp->ry, tp->rz;
    pm::Pose3d t_2_p(t_2);
    pm::Pose3d x_2_p = t_2_p.HeadToTail(x_t_p);
    pm::Pose3d::Vector6Type x_2 = x_2_p.xyzrph();

    //*x2 = x_2[0];
    //*y2 = x_2[1];
    //return true;
    return Eigen::Vector2f(x_2[0], x_2[1]);
  }

  // If we've gotten here, the given point is not part of a tracklet
  //printf("not inside\n");

  // Project (x, y) in frame 1 to frame 2
  //double x_1[6] = {pos.x(), pos.y(), 0, 0, 0, 0};

  //Pose p1 = _poses[_scan_at];
  // for numerical accuracy
  double x0 = p1.x;
  double y0 = p1.y;
  double z0 = p1.z;
  pm::Pose3d::Vector6Type x_1w;
  x_1w << p1.x-x0, p1.y-y0, p1.z-z0, p1.r, p1.p, p1.h;
  pm::Pose3d x_1w_p(x_1w);
  pm::Pose3d x_w_p = x_1w_p.HeadToTail(x_1_p);
  pm::Pose3d::Vector6Type x_w = x_w_p.xyzrph();

  //Pose p2 = _poses[_scan_at+1];
  pm::Pose3d::Vector6Type x_2w;
  x_2w << p2.x-x0, p2.y-y0, p2.z-z0, p2.r, p2.p, p2.h;
  pm::Pose3d x_2w_p(x_2w);
  pm::Pose3d x_2_p = x_2w_p.TailToTail(x_w);
  pm::Pose3d::Vector6Type x_2 = x_2_p.xyzrph();

  //printf("\t %5.3f %5.3f %5.3f\n", x_1[0], x_1[1], x_1[2]);
  //printf("\t %5.3f %5.3f %5.3f\n", x_1w[0], x_1w[1], x_1w[2]);
  //printf("\t %5.3f %5.3f %5.3f\n", x_2[0], x_2[1], x_2[2]);

  //*x2 = x_2[0];
  //*y2 = x_2[1];

  //return false;
  return Eigen::Vector2f(x_2[0], x_2[1]);
}

ObjectClass GetObjectTypeAtLocation(Tracklets *tracklets, const Eigen::Vector2f &pos, int scan_at, float res) {
  // This pose
  pm::Pose3d::Vector6Type x_1;
  x_1 << pos.x(), pos.y(), 0, 0, 0, 0;
  pm::Pose3d x_1_p(x_1);

  // Check if selected position is in any tracklets
  Tracklets::tPose *tp;
  Tracklets::tTracklet *tt;
  for (int i=0; i<tracklets->numberOfTracklets(); i++) {
    if (!tracklets->getPose(i, scan_at, tp)) {
      continue;
    }

    tt = tracklets->getTracklet(i);

    // Is this point within tracklet?
    pm::Pose3d::Vector6Type t_1;
    t_1 << tp->tx, tp->ty, tp->tz, tp->rx, tp->ry, tp->rz;
    pm::Pose3d t_1_p(t_1);
    pm::Pose3d x_t_p = t_1_p.TailToTail(x_1_p);
    pm::Pose3d::Vector6Type x_t = x_t_p.xyzrph();

    // Check if we're inside this track, otherwise this is not the track we
    // are looking for...
    if (fabs(x_t[0])<(tt->l/2 + res) && fabs(x_t[1])<(tt->w/2 + res)) {
      //printf("Inside!\n");
    } else {
      continue;
    }

    // Now project to next frame
    return StringToObjectClass(tt->objectType);
  }

  return ObjectClass::NO_OBJECT;
}

} // kitti
} // library
