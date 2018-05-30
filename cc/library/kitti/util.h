#pragma once

#include <Eigen/Core>

#include "thirdparty/perls-math-cc/pose3.h"

#include "library/kitti/tracklets.h"
#include "library/kitti/pose.h"
#include "library/kitti/object_class.h"

namespace pm = thirdparty::perls_math_cc;

namespace library {
namespace kitti {

Eigen::Vector2f FindCorrespondingPosition(Tracklets *tracklets,
                                          const Eigen::Vector2f &pos,
                                          int scan_at,
                                          int scan_des,
                                          const Pose &p1,
                                          const Pose &p2,
                                          bool *track_disappears,
                                          float res=0.0);

ObjectClass GetObjectTypeAtLocation(Tracklets *tracklets, const Eigen::Vector2f &pos, int scan_at, float res=0.0);

} // kitti
} // library
