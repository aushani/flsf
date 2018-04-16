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
  const gu::GpuData<3, float>& GetEncoding() const;
  const gu::GpuData<3, float>& GetClassification() const;

  void Apply();

 private:
  ConvolutionalLayer cl1_;
  ConvolutionalLayer cl2_;
  ConvolutionalLayer cl3_;
  ConvolutionalLayer clatent_;

  ConvolutionalLayer cl_classifier_;

  gu::GpuData<3, float> input_;

  gu::GpuData<3, float> res_cl1_;
  gu::GpuData<3, float> res_cl2_;
  gu::GpuData<3, float> res_cl3_;
  gu::GpuData<3, float> res_clatent_;
  gu::GpuData<3, float> res_classifier_;

  Network(const ConvolutionalLayer &l1, const ConvolutionalLayer &l2, const
      ConvolutionalLayer &l3, const ConvolutionalLayer &latent, const
      ConvolutionalLayer &clc);

  static std::vector<float> LoadFile(const fs::path &path);
};

} // namespace tf
} // namespace library
