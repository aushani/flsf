// Adapted from dascar
#pragma once

#include <osg/Geometry>
#include <osg/Drawable>
#include <osg/MatrixTransform>

#include "library/ray_tracing/occ_grid.h"
#include "library/flow/classification_map.h"

namespace rt = library::ray_tracing;
namespace fl = library::flow;

namespace library {
namespace ray_tracing {
namespace nodes {

//class OccGridCallback : public osg::Callback {
// public:
//  OccGridCallback();
//
//  bool run(osg::Object *object, osg::Object *data) override;
//};

class OccGrid : public osg::Group {
 public:
  OccGrid();
  OccGrid(const rt::OccGrid &og, double thresh_lo=0);
  OccGrid(const rt::OccGrid &og, const fl::ClassificationMap &cm, double thresh_lo=0);

  void Update(const rt::OccGrid &og, double thresh_lo = 0);
  void Update(const rt::OccGrid &og, const fl::ClassificationMap &cm, double thresh_lo = 0);

  void Render(bool render);
};

} // nodes
} // ray_tracing
} // library
