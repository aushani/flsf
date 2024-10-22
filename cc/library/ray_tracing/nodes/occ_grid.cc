// Adapted from dascar
#include "library/ray_tracing/nodes/occ_grid.h"

#include <iostream>

#include <osg/Geometry>
#include <osg/LineWidth>

#include "library/osg_nodes/colorful_box.h"
#include "library/kitti/object_class.h"

namespace osgn = library::osg_nodes;
namespace kt = library::kitti;

namespace library {
namespace ray_tracing {
namespace nodes {

OccGrid::OccGrid() :
 osg::Group() {
}

OccGrid::OccGrid(const rt::OccGrid &og, double thresh_lo) : osg::Group() {
  Update(og, thresh_lo);
}

OccGrid::OccGrid(const rt::OccGrid &og, const fl::ClassificationMap &cm, double thresh_lo) : osg::Group() {
  Update(og, cm, thresh_lo);
}

OccGrid::OccGrid(const rt::OccGrid &og, const fl::FilterMap &fm, double thresh_lo) : osg::Group() {
  Update(og, fm, thresh_lo);
}

void OccGrid::SetCameraCal(const kt::CameraCal &cc) {
  camera_cal_ = cc;
}

void OccGrid::Update(const rt::OccGrid &og, double thresh_lo) {
  // Remove old children
  while (getNumChildren() > 0) {
    removeChild(0, getNumChildren());
  }

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

    if (camera_cal_) {
      if (!camera_cal_->InCameraView(x, y, 0)) {
        alpha *= 0.25;
      }
    }

    osg::Vec4 color(0.1, 0.9, 0.1, alpha);
    osg::Vec3 pos(x, y, z);

    osg::ref_ptr<osgn::ColorfulBox> box = new osgn::ColorfulBox(color, pos, scale);
    addChild(box);
  }
}

void OccGrid::Update(const rt::OccGrid &og, const fl::ClassificationMap &cm, double thresh_lo) {
  // Remove old children
  while (getNumChildren() > 0) {
    removeChild(0, getNumChildren());
  }

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

    if (camera_cal_) {
      if (!camera_cal_->InCameraView(x, y, 0)) {
        alpha *= 0.25;
      }
    }

    double r = 0.3;
    double g = 0.0;
    double b = 0.3;

    if (cm.InRange(loc.i, loc.j)) {
      r = cm.GetClassProbability(loc.i, loc.j, kt::ObjectClass::CAR);
      b = cm.GetClassProbability(loc.i, loc.j, kt::ObjectClass::NO_OBJECT);
    }

    osg::Vec4 color(r, g, b, alpha);
    osg::Vec3 pos(x, y, z);

    osg::ref_ptr<osgn::ColorfulBox> box = new osgn::ColorfulBox(color, pos, scale);
    addChild(box);
  }
}

void OccGrid::Update(const rt::OccGrid &og, const fl::FilterMap &fm, double thresh_lo) {
  // Remove old children
  while (getNumChildren() > 0) {
    removeChild(0, getNumChildren());
  }

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

    if (camera_cal_) {
      if (!camera_cal_->InCameraView(x, y, 0)) {
        alpha *= 0.25;
      }
    }

    double r = 0.3;
    double g = 0.0;
    double b = 0.3;

    if (fm.InRange(loc.i, loc.j)) {
      r = fm.GetFilterProbability(loc.i, loc.j);
      b = 1 - r;
    }

    osg::Vec4 color(r, g, b, alpha);
    osg::Vec3 pos(x, y, z);

    osg::ref_ptr<osgn::ColorfulBox> box = new osgn::ColorfulBox(color, pos, scale);
    addChild(box);
  }
}

void OccGrid::Render(bool render) {
  setNodeMask(render);
}

}  // namespace nodes
}  // namespace ray_tracing
}  // namespace library
