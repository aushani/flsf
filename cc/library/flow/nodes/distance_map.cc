#include "library/flow/nodes/distance_map.h"

#include <boost/assert.hpp>
#include <osg/PolygonMode>
#include <osg/LineWidth>
#include <osg/ShapeDrawable>

#include "library/osg_nodes/composite_shape_group.h"
#include "library/kitti/object_class.h"

namespace osgn = library::osg_nodes;
namespace kt = library::kitti;

namespace library {
namespace flow {
namespace nodes {

DistanceMap::DistanceMap() :
 osg::Group() {
}

DistanceMap::DistanceMap(const fl::DistanceMap &dm, float x, float y) :
 osg::Group() {
  Update(dm, x, y);
}

void DistanceMap::Update(const fl::DistanceMap &dm, float x, float y) {
  float res = dm.GetResolution();

  int i0 = std::round(x / res);
  int j0 = std::round(y / res);

  BOOST_ASSERT(dm.InRange(i0, j0, 0, 0));

  // Get image
  osg::ref_ptr<osg::Image> im = GetImage(dm, i0, j0);

  // Now set up render
  osg::ref_ptr<osg::Texture2D> texture = new osg::Texture2D();
  texture->setResizeNonPowerOfTwoHint(false);
  texture->setImage(im);

  texture->setFilter(osg::Texture::MIN_FILTER, osg::Texture::NEAREST);
  texture->setFilter(osg::Texture::MAG_FILTER, osg::Texture::NEAREST);
  //terrain->getOrCreateStateSet()->setTextureAttribute(0, tex.get(), osg::StateAttribute::ON | osg::StateAttribute::OVERRIDE);

  osg::ref_ptr<osg::Geode> geode = new osg::Geode();


  int width = dm.MaxOffset() - dm.MinOffset() + 1;
  int height = width;

  double x0 = i0 + dm.MinOffset() - 0.5;
  double y0 = j0 + dm.MinOffset() - 0.5;
  SetUpTexture(texture, geode, x0, y0, width, height, 13);

  osg::Matrix m = osg::Matrix::identity();
  m.makeScale(res, res, res);
  //m.postMultTranslate(osg::Vec3d(x0_, y0_, -1.7)); // ground plane
  //m.postMultTranslate(osg::Vec3d(x0, y0, 0)); // ground plane

  osg::ref_ptr<osg::MatrixTransform> map_image = new osg::MatrixTransform();
  map_image->setMatrix(m);

  map_image->addChild(geode);

  // Mark origin
  osg::Vec3 pos(i0*res, j0*res, 0.0);
  osg::ref_ptr<osg::Box> box = new osg::Box(pos, res, res, 2.0);

  osg::ref_ptr<osg::ShapeDrawable> shape = new osg::ShapeDrawable(box);

  osg::ref_ptr<osgn::CompositeShapeGroup> csg = new osgn::CompositeShapeGroup();

  osg::ref_ptr<osg::StateSet> ss = csg->getOrCreateStateSet();
  osg::ref_ptr<osg::PolygonMode> pm = new osg::PolygonMode(osg::PolygonMode::FRONT_AND_BACK, osg::PolygonMode::LINE);
  osg::ref_ptr<osg::LineWidth> lw = new osg::LineWidth(8.0);
  ss->setAttributeAndModes(pm, osg::StateAttribute::ON | osg::StateAttribute::OVERRIDE);
  ss->setAttributeAndModes(lw, osg::StateAttribute::ON | osg::StateAttribute::OVERRIDE);
  csg->addChild(shape);

  // Remove old children
  while (getNumChildren() > 0) {
    removeChild(0, getNumChildren());
  }

  // Add new children
  addChild(map_image);
  addChild(csg);
}

osg::ref_ptr<osg::Image> DistanceMap::GetImage(const fl::DistanceMap &dm, int i0, int j0) {
  const int depth = 1;

  osg::ref_ptr<osg::Image> im = new osg::Image();

  int width = dm.MaxOffset() - dm.MinOffset() + 1;
  int height = width;
  im->allocateImage(width, height, depth, GL_RGBA, GL_FLOAT);

  int best_i = 0;
  int best_j = 0;
  double best_dist = -1;

  for (int i = 0; i < width; i++) {
    for (int j = 0; j < height; j++) {
      int di = dm.MinOffset() + i;
      int dj = dm.MinOffset() + j;

      double dist = dm.GetDistance(i0, j0, di, dj);

      double val = dist / kDistScaleFactor_;

      if (val > 1) val = 1;
      if (val < 0) val = 0;

      double r = val;
      double g = 0;
      double b = 1-r;

      osg::Vec4 color(r, g, b, 0.5);
      im->setColor(color, i, j, 0);

      if (dist < best_dist || best_dist < 0) {
        best_i = i;
        best_j = j;
        best_dist = dist;
      }
    }
  }

  double val = best_dist / kDistScaleFactor_;

  if (val > 1) val = 1;
  if (val < 0) val = 0;

  double r = val;
  double g = 0.5;
  double b = 1-r;

  osg::Vec4 color(r, g, b, 0.5);
  im->setColor(color, best_i, best_j, 0);

  return im;
}

void DistanceMap::SetUpTexture(osg::Texture2D *texture, osg::Geode *geode, double x0, double y0, int width, int height, int bin_num) const {
  // Adapted from dascar

  osg::ref_ptr<osg::Geometry> geometry = new osg::Geometry();
  osg::ref_ptr<osg::Vec3Array> vertices = new osg::Vec3Array;
  vertices->push_back(osg::Vec3(x0, y0, -1));
  vertices->push_back(osg::Vec3(x0+width, y0, -1));
  vertices->push_back(osg::Vec3(x0+width, y0+height, -1));
  vertices->push_back(osg::Vec3(x0, y0+height, -1));

  osg::ref_ptr<osg::DrawElementsUInt> background_indices = new osg::DrawElementsUInt(osg::PrimitiveSet::POLYGON, 0);
  background_indices->push_back(0);
  background_indices->push_back(1);
  background_indices->push_back(2);
  background_indices->push_back(3);

  osg::ref_ptr<osg::Vec4Array> colors = new osg::Vec4Array;
  colors->push_back(osg::Vec4(1.0f, 1.0f, 1.0f, 0.9f));

  osg::ref_ptr<osg::Vec2Array> texcoords = new osg::Vec2Array(4);
  (*texcoords)[0].set(0.0f,0.0f);
  (*texcoords)[1].set(1.0f,0.0f);
  (*texcoords)[2].set(1.0f,1.0f);
  (*texcoords)[3].set(0.0f,1.0f);

  geometry->setTexCoordArray(0,texcoords);
  texture->setDataVariance(osg::Object::DYNAMIC);

  osg::ref_ptr<osg::Vec3Array> normals = new osg::Vec3Array;
  normals->push_back(osg::Vec3(0.0f,0.0f,1.0f));
  geometry->setNormalArray(normals);
  geometry->setNormalBinding(osg::Geometry::BIND_OVERALL);
  geometry->addPrimitiveSet(background_indices);
  geometry->setVertexArray(vertices);
  geometry->setColorArray(colors);
  geometry->setColorBinding(osg::Geometry::BIND_OVERALL);

  geode->addDrawable(geometry);

  // Create and set up a state set using the texture from above:
  osg::ref_ptr<osg::StateSet> state_set = new osg::StateSet();
  geode->setStateSet(state_set);
  state_set->setTextureAttributeAndModes(0, texture, osg::StateAttribute::ON);

  // For this state set, turn blending on (so alpha texture looks right)
  state_set->setMode(GL_BLEND,osg::StateAttribute::ON);

  // Disable depth testing so geometry is draw regardless of depth values
  // of geometry already draw.
  state_set->setMode(GL_DEPTH_TEST,osg::StateAttribute::OFF);
  state_set->setRenderingHint( osg::StateSet::TRANSPARENT_BIN );

  // Need to make sure this geometry is draw last. RenderBins are handled
  // in numerical order so set bin number to 11 by default
  state_set->setRenderBinDetails( bin_num, "RenderBin");
}

void DistanceMap::Render(bool render) {
  setNodeMask(render);
}

} // namespace nodes
} // namespace flow
} // namespace library
