#pragma once

#include <vector>

#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;

namespace library {
namespace evaluation {

class ErrorStats {
 public:
  ErrorStats();

  void Clear();

  float GetMean() const;
  int GetNumSamples() const;

  void Process(float error);

  void WriteErrors(const fs::path &path) const;

 private:
  int count_ = 0;
  float total_error_ = 0.0;

  std::vector<float> errors_;
};

} // namespace evaluation
} // namespace library
