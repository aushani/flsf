// Adapted from dascar
#pragma once

#include <osg/Geometry>
#include <osg/Drawable>
#include <osg/MatrixTransform>

#include "library/kitti/velodyne_scan.h"
#include "library/flow/distance_map.h"

namespace fl = library::flow;

namespace library {
namespace kitti {
namespace nodes {

class PointCloud : public osg::Geometry {
 public:
  PointCloud();
  PointCloud(const VelodyneScan &scan);

  void Update(const VelodyneScan &scan);
  void Update(const VelodyneScan &scan, const fl::DistanceMap &dm, float x, float y);

  void Render(bool render);

 private:
  //static constexpr double kColorMapZMin = -2.5;
  //static constexpr double kColorMapZMax = 2.5;

  osg::ref_ptr<osg::Vec3Array> vertices_;
  osg::ref_ptr<osg::Vec4Array> colors_;
  osg::ref_ptr<osg::DrawArrays> draw_arrays_;

  //osg::ColorMap::Type cmap_ = osg::ColorMap::Type::JET;
};

} // nodes
} // kitti
} // library
