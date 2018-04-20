#include "library/flow/classification_map.h"

#include <boost/assert.hpp>

namespace library {
namespace flow {

ClassificationMap::ClassificationMap(int nx, int ny) :
 size_x_(nx),
 size_y_(ny),
 scores_(nx*ny) {
}

size_t ClassificationMap::GetIdx(int i, int j) const {
  int ii = i - MinX();
  int jj = j - MinY();

  return ii*size_y_ + jj;
}

int ClassificationMap::MinX() const {
  return -static_cast<int>(size_x_)/2;
}

int ClassificationMap::MaxX() const {
  return MinX() + size_x_ - 1;
}

int ClassificationMap::MinY() const {
  return -static_cast<int>(size_y_)/2;
}

int ClassificationMap::MaxY() const {
  return MinY() + size_y_- 1;
}

bool ClassificationMap::InRange(int i, int j) const {
  bool valid_x = i >= MinX() && i <= MaxX();
  bool valid_y = j >= MinY() && j <= MaxY();

  return valid_x && valid_y;
}

void ClassificationMap::SetClassScore(int i, int j, kt::ObjectClass c, float score) {
  BOOST_ASSERT(InRange(i, j));

  size_t idx = GetIdx(i, j);
  BOOST_ASSERT(idx < size_x_ * size_y_);

  std::map<kt::ObjectClass, float> &map = scores_[idx];
  map[c] = score;
}

float ClassificationMap::GetClassProbability(int i, int j, kt::ObjectClass c) const {
  BOOST_ASSERT(InRange(i, j));

  size_t idx = GetIdx(i, j);
  const std::map<kt::ObjectClass, float> &map = scores_[idx];

  auto class_score = map.find(c);
  if (class_score == map.end()) {
    return 0.0;
  }

  float denom = 0.0;
  for (const auto &kv : map) {
    denom += std::exp(kv.second);
  }

  return std::exp(class_score->second) / denom;
}

} // flow
} // library
