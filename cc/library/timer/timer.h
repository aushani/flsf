#pragma once

#include <chrono>

namespace library {
namespace timer {

class Timer {
 public:
  Timer();

  void Start();
  double GetMs();
  double GetSeconds();

 private:
  std::chrono::time_point<std::chrono::steady_clock> tic_;
};

}
}
