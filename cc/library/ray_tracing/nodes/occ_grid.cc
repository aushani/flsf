// Adapted from dascar
#include "library/osg_nodes/occ_grid.h"

#include <iostream>

#include <osg/Geometry>
#include <osg/LineWidth>

#include "library/osg_nodes/colorful_box.h"

namespace library {
namespace ray_tracing
namespace nodes {

OccGrid::OccGrid(const rt::OccGrid &og, double thresh_lo) : osg::Group() {
  double scale = og.GetResolution() * 0.75;

  // Iterate over occ grid and add occupied cells
  for (size_t i = 0; i < og.GetLocations().size(); i++) {
    rt::Location loc = og.GetLocations()[i];
    float val = og.GetLogOdds()[i];

    if (val <= thresh_lo) {
      continue;
    }

    double x = loc.i * og.GetResolution();
    double y = loc.j * og.GetResolution();
    double z = loc.k * og.GetResolution();

    double alpha = val*2;
    if (alpha < 0) {
      alpha = 0;
    }

    if (alpha > 0.8) {
      alpha = 0.8;
    }

    osg::Vec4 color(0.1, 0.9, 0.1, alpha);
    osg::Vec3 pos(x, y, z);

    osg::ref_ptr<ColorfulBox> box = new ColorfulBox(color, pos, scale);
    addChild(box);
  }
}

}  // namespace nodes
}  // namespace ray_tracing
}  // namespace library
