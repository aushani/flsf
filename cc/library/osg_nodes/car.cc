// adapted from dascar
#include "library/osg_nodes/car.h"

#include <iostream>

#include <osg/MatrixTransform>
#include <osg/Material>
#include <osgDB/ReadFile>

#include "library/util/angle.h"

namespace ut = library::util;

namespace library {
namespace osg_nodes {

Car::Car() : osg::MatrixTransform() {
  // read car file into osg::Node ptr
  osg::ref_ptr<osg::Node> car = osgDB::readNodeFile( _k_car_file);

  // TODO: throw an exception
  if (car == nullptr) {
    std::cerr << "error reading car file" << std::endl;
  }

  // scale and rotate car to +x, z down
  // TODO: magic numbers, specific to lexus model
  osg::Matrixd H(osg::Quat(ut::DegreesToRadians(180), osg::Vec3d(0, 0, 1)));
  H.postMultRotate(osg::Quat(ut::DegreesToRadians(-90), osg::Vec3d(1, 0, 0)));
  H.postMultScale(osg::Vec3d(_k_scale, _k_scale, _k_scale));
  H.postMultTranslate(_k_pos);
  setMatrix(H);

  // uncomment to color
  //osg::ref_ptr<osg::Material> mat = new osg::Material;
  //mat->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(0.8, 0.8, 0.8, 1.0));
  ////getOrCreateStateSet()->setAttributeAndModes(mat, osg::StateAttribute::ON);
  //getOrCreateStateSet()->setAttributeAndModes(mat, osg::StateAttribute::ON | osg::StateAttribute::OVERRIDE);

  addChild(car);

  // re-normalize normals after scaling
  osg::ref_ptr<osg::StateSet> car_stateset = car->getOrCreateStateSet();
  car_stateset->setMode(GL_NORMALIZE, osg::StateAttribute::ON);
}

} // namespace osg_nodes
} // namespace library
