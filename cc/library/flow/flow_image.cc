#include "library/flow/flow_image.h"

#include <cmath>

#include <boost/assert.hpp>

namespace library {
namespace flow {

FlowImage::FlowImage(int nx, int ny, float res) :
 size_x_(nx),
 size_y_(ny),
 resolution_(res),
 x_flow_(nx*ny, 0),
 y_flow_(nx*ny, 0),
 valid_(nx*ny, false) {
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

bool FlowImage::InRangeXY(float x, float y) const {
  int i = std::round(x / resolution_);
  int j = std::round(y / resolution_);

  return InRange(i, j);
}

float FlowImage::GetResolution() const {
  return resolution_;
}

void FlowImage::SetFlow(int i, int j, int xf, int yf) {
  BOOST_ASSERT(InRange(i, j));

  size_t idx = GetIdx(i, j);
  BOOST_ASSERT(idx < size_x_ * size_y_);
  x_flow_[idx] = xf;
  y_flow_[idx] = yf;
}

void FlowImage::SetFlowValid(int i, int j, bool valid) {
  BOOST_ASSERT(InRange(i, j));

  size_t idx = GetIdx(i, j);
  BOOST_ASSERT(idx < size_x_ * size_y_);
  valid_[idx] = valid;
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

bool FlowImage::GetFlowValid(int i, int j) const {
  BOOST_ASSERT(InRange(i, j));

  size_t idx = GetIdx(i, j);
  return valid_[idx];
}

int FlowImage::GetXFlowXY(float x, float y) const {
  int i = std::round(x / resolution_);
  int j = std::round(y / resolution_);

  return GetXFlow(i, j);
}

int FlowImage::GetYFlowXY(float x, float y) const {
  int i = std::round(x / resolution_);
  int j = std::round(y / resolution_);

  return GetYFlow(i, j);
}

bool FlowImage::GetFlowValidXY(float x, float y) const {
  int i = std::round(x / resolution_);
  int j = std::round(y / resolution_);

  return GetFlowValid(i, j);
}

} // flow
} // library
