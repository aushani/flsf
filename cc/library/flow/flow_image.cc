#include "library/flow/flow_image.h"

#include <boost/assert.hpp>

namespace library {
namespace flow {

FlowImage::FlowImage(int nx, int ny) :
 size_x_(nx),
 size_y_(ny),
 x_flow_(nx*ny, 0),
 y_flow_(nx*ny, 0) {
}

size_t FlowImage::GetIdx(int i, int j) const {
  int ii = i - MinX();
  int jj = j - MinY();

  return ii*size_y_ + jj;
}

int FlowImage::MinX() const {
  return -static_cast<int>(size_x_)/2;
}

int FlowImage::MaxX() const {
  return MinX() + size_x_ - 1;
}

int FlowImage::MinY() const {
  return -static_cast<int>(size_y_)/2;
}

int FlowImage::MaxY() const {
  return MinY() + size_y_- 1;
}

bool FlowImage::InRange(int i, int j) const {
  bool valid_x = i >= MinX() && i <= MaxX();
  bool valid_y = j >= MinY() && j <= MaxY();

  return valid_x && valid_y;
}

void FlowImage::SetFlow(int i, int j, int xf, int yf) {
  BOOST_ASSERT(InRange(i, j));

  size_t idx = GetIdx(i, j);
  BOOST_ASSERT(idx < size_x_ * size_y_);
  x_flow_[idx] = xf;
  y_flow_[idx] = yf;
}

int FlowImage::GetXFlow(int i, int j) const {
  BOOST_ASSERT(InRange(i, j));

  size_t idx = GetIdx(i, j);
  return x_flow_[idx];
}

int FlowImage::GetYFlow(int i, int j) const {
  BOOST_ASSERT(InRange(i, j));

  size_t idx = GetIdx(i, j);
  return y_flow_[idx];
}

} // flow
} // library
