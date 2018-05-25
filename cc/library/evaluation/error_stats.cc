#include "library/evaluation/error_stats.h"

namespace library {
namespace evaluation {

ErrorStats::ErrorStats() {
  Clear();
}

void ErrorStats::Clear() {
  total_error_ = 0.0;
  count_ = 0;

  errors_.clear();
}

float ErrorStats::GetMean() const {
  if (count_ == 0) {
    return 0.0;
  }

  return total_error_ / count_;
}

int ErrorStats::GetNumSamples() const {
  return count_;
}

void ErrorStats::Process(float error) {
  errors_.push_back(error);

  count_++;
  total_error_ += error;
}

void ErrorStats::WriteErrors(const fs::path &path) const {
  std::ofstream file(path.string());

  for (const float err : errors_) {
    file << err << std::endl;
  }
}

} // namespace evaluation
} // namespace library
