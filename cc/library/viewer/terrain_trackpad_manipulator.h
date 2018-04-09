// Adapted from dascar
#pragma once

#include <osgGA/TerrainManipulator>

namespace library {
namespace viewer {

class TerrainTrackpadManipulator : public osgGA::TerrainManipulator {
 public:
  TerrainTrackpadManipulator(int flags = DEFAULT_SETTINGS);
  bool performMovement() override;
};

} // namespace viewer
} // namespace library
