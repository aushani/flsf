#include "library/flow/distance_map.h"

#include <boost/assert.hpp>

namespace library {
namespace flow {

DistanceMap::DistanceMap(int nx, int ny, int window_size, float res) :
 size_x_(nx),
 size_y_(ny),
 window_size_(window_size),
 resolution_(res),
 distances_(nx*ny*window_size*window_size) {
}

size_t DistanceMap::GetIdx(int i, int j, int di, int dj) const {
  int ii = i - MinX();
  int jj = j - MinY();

  int kk = di - MinOffset();
  int ll = dj - MinOffset();

  return ((ii*size_y_ + jj) * window_size_ + kk) * window_size_ + ll;
}

int DistanceMap::MinX() const {
  return -static_cast<int>(size_x_)/2;
}

int DistanceMap::MaxX() const {
  return MinX() + size_x_ - 1;
}

int DistanceMap::MinY() const {
  return -static_cast<int>(size_y_)/2;
}

int DistanceMap::MaxY() const {
  return MinY() + size_y_- 1;
}

int DistanceMap::MinOffset() const {
  return -static_cast<int>(window_size_)/2;
}

int DistanceMap::MaxOffset() const {
  return MinOffset() + window_size_ - 1;
}

float DistanceMap::GetResolution() const {
  return resolution_;
}

bool DistanceMap::InRange(int i, int j, int di, int dj) const {
  bool valid_x = i >= MinX() && i <= MaxX();
  bool valid_y = j >= MinY() && j <= MaxY();

  bool valid_di = di >= MinOffset() && di <= MaxOffset();
  bool valid_dj = dj >= MinOffset() && dj <= MaxOffset();

  return valid_x && valid_y && valid_di && valid_dj;
}

void DistanceMap::SetDistance(int i, int j, int di, int dj, float dist) {
  BOOST_ASSERT(InRange(i, j, di, dj));

  size_t idx = GetIdx(i, j, di, dj);
  BOOST_ASSERT(idx < distances_.size());

  distances_[idx] = dist;
}

float DistanceMap::GetDistance(int i, int j, int di, int dj) const {
  BOOST_ASSERT(InRange(i, j, di, dj));

  size_t idx = GetIdx(i, j, di, dj);
  return distances_[idx];
}

float DistanceMap::GetDistanceXY(float x, float y, float dx, float dy) const {
  int i = std::round(x / resolution_);
  int j = std::round(y / resolution_);

  int di = std::round(dx / resolution_);
  int dj = std::round(dy / resolution_);

  return GetDistance(i, j, di, dj);
}

} // flow
} // library
