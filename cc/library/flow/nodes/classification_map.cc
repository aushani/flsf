#include "library/flow/nodes/classification_map.h"

namespace library {
namespace flow {
namespace nodes {

ClassificationMap::ClassificationMap(const fl::ClassificationMap &cm, float res) :
 osg::Group() {
  // Get image
  osg::ref_ptr<osg::Image> im = GetImage(cm);

  // Now set up render
  osg::ref_ptr<osg::Texture2D> texture = new osg::Texture2D();
  texture->setResizeNonPowerOfTwoHint(false);
  texture->setImage(im);

  texture->setFilter(osg::Texture::MIN_FILTER, osg::Texture::NEAREST);
  texture->setFilter(osg::Texture::MAG_FILTER, osg::Texture::NEAREST);
  //terrain->getOrCreateStateSet()->setTextureAttribute(0, tex.get(), osg::StateAttribute::ON | osg::StateAttribute::OVERRIDE);

  osg::ref_ptr<osg::Geode> geode = new osg::Geode();

  double x0 = cm.MinX();
  double y0 = cm.MinY();
  int width = cm.MaxX() - cm.MinX() + 1;
  int height = cm.MaxY() - cm.MinY() + 1;

  SetUpTexture(texture, geode, x0, y0, width, height, 12);

  osg::Matrix m = osg::Matrix::identity();
  m.makeScale(res, res, res);
  //m.postMultTranslate(osg::Vec3d(x0_, y0_, -1.7)); // ground plane
  //m.postMultTranslate(osg::Vec3d(x0, y0, 0)); // ground plane

  osg::ref_ptr<osg::MatrixTransform> map_image = new osg::MatrixTransform();
  map_image->setMatrix(m);

  // Ready to add
  map_image->addChild(geode);
  addChild(map_image);
}

osg::ref_ptr<osg::Image> ClassificationMap::GetImage(const fl::ClassificationMap &cm) {
  const int depth = 1;

  osg::ref_ptr<osg::Image> im = new osg::Image();

  int width = cm.MaxX() - cm.MinX() + 1;
  int height = cm.MaxY() - cm.MinY() + 1;
  im->allocateImage(width, height, depth, GL_RGBA, GL_FLOAT);

  for (int i = 0; i < width; i++) {
    for (int j = 0; j < height; j++) {
      int ii = cm.MinX() + i;
      int jj = cm.MinY() + j;

      double p_car = cm.GetClassProbability(ii, jj, ObjectClass::CAR);

      double r = p_car;
      double g = 0;
      double b = 1-r;

      osg::Vec4 color(r, g, b, 0.5);
      im->setColor(color, i, j, 0);
    }
  }

  return im;
}

void ClassificationMap::SetUpTexture(osg::Texture2D *texture, osg::Geode *geode, double x0, double y0, int width, int height, int bin_num) const {
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

} // namespace nodes
} // namespace flow
} // namespace library
