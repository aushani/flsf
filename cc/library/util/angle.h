#pragma once

#include <math.h>

namespace library {
namespace util {

inline double MinimizeAngle(double t) {
  while (t < -M_PI) t += 2*M_PI;
  while (t >  M_PI) t -= 2*M_PI;
  return t;
}

inline double DegreesToRadians(double d) {
  return d * M_PI / 180.0;
}

inline double RadiansToDegrees(double r) {
  return r * 180.0 / M_PI;
}

} // namespace util
} // namespace library
