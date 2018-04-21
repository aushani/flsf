#include "library/kitti/object_class.h"

#include <boost/assert.hpp>

namespace library {
namespace kitti {

ObjectClass IntToObjectClass(int x) {
  switch(x) {
    case 0:
      return ObjectClass::CAR;

    case 1:
      return ObjectClass::CYCLIST;

    case 2:
      return ObjectClass::MISC;

    case 3:
      return ObjectClass::NO_OBJECT;

    case 4:
      return ObjectClass::PEDESTRIAN;

    case 5:
      return ObjectClass::TRAM;

    case 6:
      return ObjectClass::TRUCK;

    case 7:
      return ObjectClass::VAN;

    case 8:
      return ObjectClass::PERSON_SITTING;

    default:
      // Show have gotten something by now
      BOOST_ASSERT(false);
  }

  return ObjectClass::NO_OBJECT;
}

int ObjectClassToInt(const ObjectClass &c) {
  switch(c) {
    case ObjectClass::CAR:
      return 0;

    case ObjectClass::CYCLIST:
      return 1;

    case ObjectClass::MISC:
      return 2;

    case ObjectClass::NO_OBJECT:
      return 3;

    case ObjectClass::PEDESTRIAN:
      return 4;

    case ObjectClass::TRAM:
      return 5;

    case ObjectClass::TRUCK:
      return 6;

    case ObjectClass::VAN:
      return 7;

    case ObjectClass::PERSON_SITTING:
      return 8;

    default:
      // Show have gotten something by now
      BOOST_ASSERT(false);
  }

  return -1;
}

ObjectClass StringToObjectClass(const std::string &s) {
  if (s == "Car") {
    return ObjectClass::CAR;
  }

  if (s == "Cyclist") {
    return ObjectClass::CYCLIST;
  }

  if (s == "Misc") {
    return ObjectClass::MISC;
  }

  if (s == "NoObject") {
    return ObjectClass::NO_OBJECT;
  }

  if (s == "Pedestrian") {
    return ObjectClass::PEDESTRIAN;
  }

  if (s == "Tram") {
    return ObjectClass::TRAM;
  }

  if (s == "Truck") {
    return ObjectClass::TRUCK;
  }

  if (s == "Van") {
    return ObjectClass::VAN;
  }

  if (s == "Person (sitting)") {
    return ObjectClass::PERSON_SITTING;
  }

  // Should have gotten something by now
  BOOST_ASSERT(false);
}

std::string ObjectClassToString(const ObjectClass &c) {
  switch(c) {
    case ObjectClass::CAR:
      return "Car";

    case ObjectClass::CYCLIST:
      return "Cyclist";

    case ObjectClass::MISC:
      return "Misc";

    case ObjectClass::NO_OBJECT:
      return "NoObject";

    case ObjectClass::PEDESTRIAN:
      return "Pedestrian";

    case ObjectClass::TRAM:
      return "Tram";

    case ObjectClass::TRUCK:
      return "Truck";

    case ObjectClass::VAN:
      return "Van";

    case ObjectClass::PERSON_SITTING:
      return "Person (sitting)";

    default:
      // Show have gotten something by now
      BOOST_ASSERT(false);
  }

  return "";
}

} // kitti
} // library
