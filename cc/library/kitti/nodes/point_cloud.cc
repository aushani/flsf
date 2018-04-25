// Adpated from dascar
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
  state->setAttribute(new osg::Point(1), osg::StateAttribute::ON);
}

void PointCloud::Render(bool render) {
  setNodeMask(render);
}

} // nodes
} // kitti
} // library
