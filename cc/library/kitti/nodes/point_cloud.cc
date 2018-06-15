// Adpatepd from dascar
#include "library/kitti/nodes/point_cloud.h"

#include <osg/Point>

namespace library {
namespace kitti {
namespace nodes {

PointCloud::PointCloud() :
 osg::Geometry(),
 vertices_(new osg::Vec3Array),
 colors_(new osg::Vec4Array) {

}

PointCloud::PointCloud(const VelodyneScan &scan) :
 osg::Geometry(),
 vertices_(new osg::Vec3Array),
 colors_(new osg::Vec4Array) {
  Update(scan);
}

void PointCloud::Update(const VelodyneScan &scan) {
  // Remove previous sets
  while (getNumPrimitiveSets() > 0) {
    removePrimitiveSet(0, getNumPrimitiveSets());
  }

  // Clear arrays
  vertices_->resizeArray(0);
  colors_->resizeArray(0);

  for (const auto &hit : scan.GetHits()) {
    vertices_->push_back(osg::Vec3(hit.x(), hit.y(), hit.z()));
    //double z = hit.z();
    //double c = 0;
    //if (z < kColorMapZMin) {
    //  c = 0.0;
    //} else if (z > kColorMapZMax) {
    //  c = 1.0;
    //} else {
    //  c = (z - kColorMapZMin)/(kColorMapZMax - kColorMapZMin);
    //}
    //colors_->push_back(osg::Vec4(1-c, 0, c, 0));
    colors_->push_back(osg::Vec4(0.2, 0.2, 0.2, 0));
  }

  setVertexArray(vertices_);
  setColorArray(colors_);
  setColorBinding(osg::Geometry::BIND_PER_VERTEX);

  draw_arrays_ = new osg::DrawArrays(osg::PrimitiveSet::POINTS, 0, vertices_->size());
  addPrimitiveSet(draw_arrays_);

  //_geode->addDrawable(this);
  osg::ref_ptr<osg::StateSet> state = getOrCreateStateSet();
  state->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
  state->setAttribute(new osg::Point(8), osg::StateAttribute::ON);
}

void PointCloud::Update(const VelodyneScan &scan, const fl::DistanceMap &dm, float x, float y) {
  // Remove previous sets
  while (getNumPrimitiveSets() > 0) {
    removePrimitiveSet(0, getNumPrimitiveSets());
  }

  // Clear arrays
  vertices_->resizeArray(0);
  colors_->resizeArray(0);

  for (const auto &hit : scan.GetHits()) {
    vertices_->push_back(osg::Vec3(hit.x(), hit.y(), hit.z()));

    float dx = hit.x() - x;
    float dy = hit.y() - y;

    double r = 0.2;
    double g = 0.2;
    double b = 0.2;

    if (dm.InRangeXY(hit.x(), hit.y(), dx, dy)) {
      float dist = dm.GetDistanceXY(hit.x(), hit.y(), dx, dy);

      double val = dist / 2.0;

      if (val > 1) val = 1;
      if (val < 0) val = 0;

      r = val;
      g = 1-r;
      b = 0;
    }

    colors_->push_back(osg::Vec4(r, g, b, 0));
  }

  setVertexArray(vertices_);
  setColorArray(colors_);
  setColorBinding(osg::Geometry::BIND_PER_VERTEX);

  draw_arrays_ = new osg::DrawArrays(osg::PrimitiveSet::POINTS, 0, vertices_->size());
  addPrimitiveSet(draw_arrays_);

  //_geode->addDrawable(this);
  osg::ref_ptr<osg::StateSet> state = getOrCreateStateSet();
  state->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
  state->setAttribute(new osg::Point(4), osg::StateAttribute::ON);
}

void PointCloud::Render(bool render) {
  setNodeMask(render);
}

} // nodes
} // kitti
} // library
