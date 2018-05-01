#pragma once

#include <osg/Geometry>
#include <osg/Drawable>
#include <osg/MatrixTransform>
#include <osg/Texture2D>
#include <osg/Geode>

#include "library/flow/filter_map.h"

namespace fl = library::flow;

namespace library {
namespace flow {
namespace nodes {

class FilterMap : public osg::Group {
 public:
  FilterMap();
  FilterMap(const fl::FilterMap &fm);

  void Update(const fl::FilterMap &fm);

  void Render(bool render);

 private:
  osg::ref_ptr<osg::Image> GetImage(const fl::FilterMap &fm);
  void SetUpTexture(osg::Texture2D *texture, osg::Geode *geode, double x0, double y0, int width, int height, int bin_num) const;
};

} // nodes
} // flow
} // library
