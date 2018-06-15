#pragma once

#include <cstddef>
#include <cmath>
#include <vector>
#include <map>

namespace library {
namespace flow {

class DistanceMap {
 public:
  DistanceMap(int nx, int ny, int window_size, float res);

  void SetDistance(int i, int j, int di, int dj, float dist);

  int MinX() const;
  int MaxX() const;
  int MinY() const;
  int MaxY() const;

  int MinOffset() const;
  int MaxOffset() const;
  bool InRange(int i, int j, int di, int dj) const;
  bool InRangeXY(float x, float y, float dx, float dy) const;

  float GetResolution() const;

  float GetDistance(int i, int j, int di, int dj) const;
  float GetDistanceXY(float x, float y, float dx, float dy) const;

 private:
  size_t size_x_;
  size_t size_y_;
  size_t window_size_;

  float resolution_;

  std::vector<float> distances_;

  size_t GetIdx(int i, int j, int di, int dj) const;
};

} // flow
} // library
