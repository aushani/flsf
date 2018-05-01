#include "library/flow/filter_map.h"

#include <boost/assert.hpp>

namespace library {
namespace flow {

FilterMap::FilterMap(int nx, int ny, float res) :
 size_x_(nx),
 size_y_(ny),
 resolution_(res),
 probs_(nx*ny) {
}

size_t FilterMap::GetIdx(int i, int j) const {
  int ii = i - MinX();
  int jj = j - MinY();

  return ii*size_y_ + jj;
}

int FilterMap::MinX() const {
  return -static_cast<int>(size_x_)/2;
}

int FilterMap::MaxX() const {
  return MinX() + size_x_ - 1;
}

int FilterMap::MinY() const {
  return -static_cast<int>(size_y_)/2;
}

int FilterMap::MaxY() const {
  return MinY() + size_y_- 1;
}

float FilterMap::GetResolution() const {
  return resolution_;
}

bool FilterMap::InRange(int i, int j) const {
  bool valid_x = i >= MinX() && i <= MaxX();
  bool valid_y = j >= MinY() && j <= MaxY();

  return valid_x && valid_y;
}

void FilterMap::SetFilterProbability(int i, int j, float prob) {
  BOOST_ASSERT(InRange(i, j));

  size_t idx = GetIdx(i, j);
  probs_[idx] = prob;
}

float FilterMap::GetFilterProbability(int i, int j) const {
  BOOST_ASSERT(InRange(i, j));

  size_t idx = GetIdx(i, j);
  return probs_[idx];
}

float FilterMap::GetFilterProbabilityXY(float x, float y) const {
  int i = std::round(x / resolution_);
  int j = std::round(y / resolution_);

  return GetFilterProbability(i, j);
}

} // flow
} // library
