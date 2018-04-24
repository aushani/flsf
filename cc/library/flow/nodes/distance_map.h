#pragma once

#include <osg/Geometry>
#include <osg/Drawable>
#include <osg/MatrixTransform>
#include <osg/Texture2D>
#include <osg/Geode>

#include "library/flow/distance_map.h"

namespace fl = library::flow;

namespace library {
namespace flow {
namespace nodes {

class DistanceMap : public osg::Group {
 public:
  DistanceMap(const fl::DistanceMap &dm, float x, float y);

 private:
  osg::ref_ptr<osg::Image> GetImage(const fl::DistanceMap &cm, int i0, int j0);
  void SetUpTexture(osg::Texture2D *texture, osg::Geode *geode, double x0, double y0, int width, int height, int bin_num) const;
};

} // nodes
} // flow
} // library
