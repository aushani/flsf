#include "timer.h"

namespace library {
namespace timer {

Timer::Timer() :
  tic_(std::chrono::steady_clock::now()) {
}

void Timer::Start() {
  tic_ = std::chrono::steady_clock::now();
}

double Timer::GetMs() {
  auto toc = std::chrono::steady_clock::now();
  auto t_us = std::chrono::duration_cast<std::chrono::microseconds>(toc - tic_);
  return t_us.count()/1000.0;
}

double Timer::GetSeconds() {
  auto toc = std::chrono::steady_clock::now();
  auto t_us = std::chrono::duration_cast<std::chrono::microseconds>(toc - tic_);
  return t_us.count()/(1000.0 * 1000.0);
}

}
}
