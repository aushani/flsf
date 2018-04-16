#pragma once

#include <cstddef>
#include <cmath>
#include <vector>
#include <map>

namespace library {
namespace flow {

enum class ObjectClass {
  CAR,
  CYCLIST,
  MISC,
  NO_OBJECT,
  PEDESTRIAN,
  TRAM,
  TRUCK,
  VAN
};

class ClassificationMap {
 public:
  ClassificationMap(int nx, int ny);

  void SetClassScore(int i, int j, ObjectClass c, float score);

  int MinX() const;
  int MaxX() const;
  int MinY() const;
  int MaxY() const;
  bool InRange(int i, int j) const;

  float GetClassProbability(int i, int j, ObjectClass c) const;

  static ObjectClass IntToObjectClass(int x);

 private:
  size_t size_x_;
  size_t size_y_;

  std::vector< std::map<ObjectClass, float> > scores_;

  size_t GetIdx(int i, int j) const;
};

} // flow
} // library
