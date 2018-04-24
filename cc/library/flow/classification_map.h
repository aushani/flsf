#pragma once

#include <cstddef>
#include <cmath>
#include <vector>
#include <map>

#include "library/kitti/object_class.h"

namespace kt = library::kitti;

namespace library {
namespace flow {

class ClassificationMap {
 public:
  ClassificationMap(int nx, int ny, float res);

  void SetClassScore(int i, int j, kt::ObjectClass c, float score);

  int MinX() const;
  int MaxX() const;
  int MinY() const;
  int MaxY() const;
  bool InRange(int i, int j) const;

  float GetResolution() const;

  float GetClassProbability(int i, int j, kt::ObjectClass c) const;

 private:
  size_t size_x_;
  size_t size_y_;

  float resolution_;

  std::vector< std::map<kt::ObjectClass, float> > scores_;

  size_t GetIdx(int i, int j) const;
};

} // flow
} // library
