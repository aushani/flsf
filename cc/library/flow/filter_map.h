#pragma once

#include <cstddef>
#include <cmath>
#include <vector>

namespace library {
namespace flow {

class FilterMap {
 public:
  FilterMap(int nx, int ny, float res);

  void SetFilterProbability(int i, int j, float prob);

  int MinX() const;
  int MaxX() const;
  int MinY() const;
  int MaxY() const;
  bool InRange(int i, int j) const;

  float GetResolution() const;

  float GetFilterProbability(int i, int j) const;
  float GetFilterProbabilityXY(float x, float y) const;

 private:
  size_t size_x_;
  size_t size_y_;

  float resolution_;

  std::vector<float> probs_;

  size_t GetIdx(int i, int j) const;
};

} // flow
} // library
