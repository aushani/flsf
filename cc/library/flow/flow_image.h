#pragma once

#include <cstddef>
#include <vector>

namespace library {
namespace flow {

class FlowImage {
 public:
  FlowImage(int nx, int ny, float res);

  void SetFlow(int i, int j, int xf, int yf);
  void SetFlowValid(int i, int j, bool valid);

  int MinX() const;
  int MaxX() const;
  int MinY() const;
  int MaxY() const;
  bool InRange(int i, int j) const;

  float GetResolution() const;

  int GetXFlow(int i, int j) const;
  int GetYFlow(int i, int j) const;
  bool GetFlowValid(int i, int j) const;

  int GetXFlowXY(float x, float y) const;
  int GetYFlowXY(float x, float y) const;
  bool GetFlowValidXY(float x, float y) const;

 private:
  size_t size_x_;
  size_t size_y_;

  float resolution_;

  std::vector<int> x_flow_;
  std::vector<int> y_flow_;
  std::vector<bool> valid_;

  size_t GetIdx(int i, int j) const;
};

} // flow
} // library
