// Adapted from dascar
#pragma once

#include <osg/MatrixTransform>
#include <osg/Vec4>

#include "library/kitti/tracklets.h"

namespace kt = library::kitti;

namespace library {
namespace kitti {
namespace nodes {

class Tracklets : public osg::Group {
 public:
  Tracklets();
  Tracklets(kt::Tracklets *tracklets, int frame);

  void Update(kt::Tracklets *tracklets, int frame);

  void Render(bool render);

 private:
  osg::Vec4 color_car_        = osg::Vec4(1.0, 0.0, 0.0, 0.8);
  osg::Vec4 color_pedestrian_ = osg::Vec4(0.0, 0.0, 1.0, 0.8);
  osg::Vec4 color_cyclist_    = osg::Vec4(0.0, 0.0, 1.0, 0.8);

  osg::Vec4 color_other_      = osg::Vec4(1.0, 0.0, 1.0, 0.8);
};

}  // namespace nodes
}  // namespace kitti
}  // namespace library
