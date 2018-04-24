#pragma once

#include <cstddef>
#include <cmath>
#include <vector>
#include <map>

namespace library {
namespace flow {

class DistanceMap {
 public:
  DistanceMap(int nx, int ny, int window_size);

  void SetDistance(int i, int j, int di, int dj, float dist);

  int MinX() const;
  int MaxX() const;
  int MinY() const;
  int MaxY() const;

  int MinOffset() const;
  int MaxOffset() const;
  bool InRange(int i, int j, int di, int dj) const;

  float GetDistance(int i, int j, int di, int dj) const;

 private:
  size_t size_x_;
  size_t size_y_;
  size_t window_size_;

  std::vector<float> distances_;

  size_t GetIdx(int i, int j, int di, int dj) const;
};

} // flow
} // library
