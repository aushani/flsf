#include "library/kitti/object_class.h"

#include <boost/assert.hpp>

namespace library {
namespace kitti {

ObjectClass IntToObjectClass(int x) {
  ObjectClass c = ObjectClass::NO_OBJECT;

  switch(x) {
    case 0:
      c = ObjectClass::CAR;
      break;

    case 1:
      c = ObjectClass::CYCLIST;
      break;

    case 2:
      c = ObjectClass::MISC;
      break;

    case 3:
      c = ObjectClass::NO_OBJECT;
      break;

    case 4:
      c = ObjectClass::PEDESTRIAN;
      break;

    case 5:
      c = ObjectClass::TRAM;
      break;

    case 6:
      c = ObjectClass::TRUCK;
      break;

    case 7:
      c = ObjectClass::VAN;
      break;

    case 8:
      c = ObjectClass::PERSON_SITTING;
      break;

    default:
      // Show have gotten something by now
      BOOST_ASSERT(false);
      break;
  }

  return c;
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

  if (s == "NO_OBJECT") {
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
      return "NO_OBJECT";

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
