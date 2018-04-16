#pragma once

#include <osg/Geometry>
#include <osg/Drawable>
#include <osg/MatrixTransform>

#include "library/flow/flow_image.h"

namespace fl = library::flow;

namespace library {
namespace flow {
namespace nodes {

class FlowImage : public osg::Group {
 public:
  FlowImage(const fl::FlowImage &fi, float res);
};

} // nodes
} // flow
} // library
