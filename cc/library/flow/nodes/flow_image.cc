#include "library/flow/nodes/flow_image.h"

#include <iostream>

#include <osg/Geometry>
#include <osg/LineWidth>

namespace library {
namespace flow {
namespace nodes {

FlowImage::FlowImage() :
 osg::Group() {
}

FlowImage::FlowImage(const fl::FlowImage &fi) :
 osg::Group() {
  Update(fi);
}

void FlowImage::Update(const fl::FlowImage &fi) {
  // Remove old children
  while (getNumChildren() > 0) {
    removeChild(0, getNumChildren());
  }

  float z = 0;
  osg::Vec4 color(0, 0, 0, 0);

  for (int i=fi.MinX(); i<=fi.MaxX(); i++) {
    for (int j=fi.MinY(); j<=fi.MaxY(); j++) {
      // Check for valid flow
      if (!fi.GetFlowValid(i, j)) {
        continue;
      }

      float fx = fi.GetResolution() * fi.GetXFlow(i, j);
      float fy = fi.GetResolution() * fi.GetYFlow(i, j);

      float x = i * fi.GetResolution();
      float y = j * fi.GetResolution();

      osg::Vec3 sp(x, y, z);
      osg::Vec3 ep(x+fx*0.95, y+fy*0.95, z);

      osg::ref_ptr<osg::Vec3Array> vertices = new osg::Vec3Array();

      // set vertices
      vertices->push_back(sp);
      vertices->push_back(ep);

      osg::ref_ptr<osg::Geometry> geometry = new osg::Geometry();
      osg::ref_ptr<osg::DrawElementsUInt> line =
              new osg::DrawElementsUInt(osg::PrimitiveSet::LINES, 0);
      line->push_back(0);
      line->push_back(1);
      geometry->addPrimitiveSet(line);

      osg::ref_ptr<osg::LineWidth> linewidth = new osg::LineWidth(2.0);
      geometry->getOrCreateStateSet()->setAttribute(linewidth);

      // turn off lighting
      geometry->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF | osg::StateAttribute::OVERRIDE);

      osg::ref_ptr<osg::Vec4Array> colors = new osg::Vec4Array();
      colors->push_back(color);
      geometry->setColorArray(colors);
      geometry->setColorBinding(osg::Geometry::BIND_PER_PRIMITIVE_SET);

      geometry->setVertexArray(vertices);

      addChild(geometry);
    }
  }
}

void FlowImage::Render(bool render) {
  setNodeMask(render);
}

}  // namespace nodes
}  // namespace flow
}  // namespace library
