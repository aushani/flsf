#pragma once

#include <osg/Geometry>
#include <osg/Drawable>
#include <osg/MatrixTransform>
#include <osg/Texture2D>
#include <osg/Geode>

#include "library/flow/classification_map.h"

namespace fl = library::flow;

namespace library {
namespace flow {
namespace nodes {

class ClassificationMap : public osg::Group {
 public:
  ClassificationMap();
  ClassificationMap(const fl::ClassificationMap &cm);

  void Update(const fl::ClassificationMap &cm);

  void Render(bool render);

 private:
  osg::ref_ptr<osg::Image> GetImage(const fl::ClassificationMap &cm);
  void SetUpTexture(osg::Texture2D *texture, osg::Geode *geode, double x0, double y0, int width, int height, int bin_num) const;
};

} // nodes
} // flow
} // library
