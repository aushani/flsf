#pragma once

#include <string>
#include <vector>

#include <boost/filesystem.hpp>

#include "library/gpu_util/gpu_data.cu.h"
#include "library/tf/convolutional_layer.cu.h"
#include "library/ray_tracing/occ_grid.h"

namespace fs = boost::filesystem;
namespace gu = library::gpu_util;
namespace rt = library::ray_tracing;

namespace library {
namespace tf {

// Forward declaration of data
typedef struct NetworkData NetworkData;

class Network {
 public:
  static Network LoadNetwork(const fs::path &path);

  void SetInput(const rt::OccGrid &og);

  void Apply(gu::GpuData<3, float> *encoding, gu::GpuData<2, float> *p_background);

  int GetEncodingDim() const;

 private:
  gu::GpuData<3, float> input_;

  std::vector<ConvolutionalLayer> encoder_;
  std::vector< gu::GpuData<3, float> > intermediate_encoder_results_;

  ConvolutionalLayer filter_;
  gu::GpuData<3, float> filter_res_;

  Network(const std::vector<ConvolutionalLayer> &encoder, const ConvolutionalLayer &filter);

  static std::vector<float> LoadFile(const fs::path &path);
  static std::vector<int> LoadDimFile(const fs::path &path);

  void ComputeFilterProbability(gu::GpuData<2, float> *p_background);
};

} // namespace tf
} // namespace library
