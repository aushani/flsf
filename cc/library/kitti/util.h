#pragma once

#include <Eigen/Core>

#include "thirdparty/perls-math-cc/pose3.h"

#include "library/kitti/tracklets.h"
#include "library/kitti/pose.h"

namespace pm = thirdparty::perls_math_cc;

namespace library {
namespace kitti {

Eigen::Vector2f FindCorrespondingPosition(Tracklets *tracklets,
                                          const Eigen::Vector2f &pos,
                                          int scan_at,
                                          int scan_des,
                                          const Pose &p1,
                                          const Pose &p2);

} // kitti
} // library
