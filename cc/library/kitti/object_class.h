#pragma once

#include <string>

namespace library {
namespace kitti {

enum class ObjectClass {
  CAR,
  CYCLIST,
  MISC,
  NO_OBJECT,
  PEDESTRIAN,
  TRAM,
  TRUCK,
  VAN,
  PERSON_SITTING
};

ObjectClass IntToObjectClass(int x);
ObjectClass StringToObjectClass(const std::string &s);
std::string ObjectClassToString(const ObjectClass &c);

} // kitti
} // library
