#include "library/tf/network.h"

#include <boost/assert.hpp>

#include "library/ray_tracing/occ_grid.h"
#include "library/gpu_util/gpu_data.cu.h"

#include "library/tf/convolutional_layer.h"

namespace gu = library::gpu_util;

namespace library {
namespace tf {

struct NetworkData {
  ConvolutionalLayer cl1;
  ConvolutionalLayer cl2;
  ConvolutionalLayer cl3;
  ConvolutionalLayer clatent;

  ConvolutionalLayer cl_classifier;

  gu::GpuData<3> input;

  gu::GpuData<3> res_cl1;
  gu::GpuData<3> res_cl2;
  gu::GpuData<3> res_cl3;
  gu::GpuData<3> res_clatent;
  gu::GpuData<3> res_classifier;

  NetworkData(const ConvolutionalLayer &l1, const ConvolutionalLayer &l2, const ConvolutionalLayer &l3, const ConvolutionalLayer &latent, const ConvolutionalLayer &clc) :
   cl1(l1), cl2(l2), cl3(l3), clatent(latent),
   cl_classifier(clc),
   input(167, 167, 14),
   res_cl1(167, 167, 200),
   res_cl2(167, 167, 100),
   res_cl3(167, 167, 50),
   res_clatent(167, 167, 25),
   res_classifier(167, 167, 8) {
    input.SetCoalesceDim(0);
    res_cl1.SetCoalesceDim(0);
    res_cl2.SetCoalesceDim(0);
    res_cl3.SetCoalesceDim(0);
    res_clatent.SetCoalesceDim(0);
    res_classifier.SetCoalesceDim(0);
  }
};

Network::Network(NetworkData *data) :
 data_(data) {
}

__global__ void CopyOccGrid(const gu::GpuData<1, rt::Location> locations, const
    gu::GpuData<1, float> log_odds, gu::GpuData<3> dense) {
  // Figure out which hit this thread is processing
  const int bidx = blockIdx.x;
  const int tidx = threadIdx.x;
  const int threads = blockDim.x;

  const int idx = tidx + bidx * threads;
  if (idx >= locations.Size()) {
    return;
  }

  const auto &loc = locations(idx);
  float lo = log_odds(idx);

  float p = 1.0 / (1.0 + expf(-lo));

  int i = loc.i - dense.GetDim(0)/2;
  int j = loc.j - dense.GetDim(1)/2;
  int k = loc.k - dense.GetDim(2)/2;

  if (i < 0 || i >= dense.GetDim(0)) {
    return;
  }

  if (j < 0 || j >= dense.GetDim(1)) {
    return;
  }

  if (k < 0 || k >= dense.GetDim(2)) {
    return;
  }

  dense(i, j, k) = p;
}

void Network::SetInput(const rt::OccGrid &og) {
  // Get size
  int sz = og.GetLocations().size();

  // Make GpuData objects
  gu::GpuData<1, rt::Location> locations(sz);
  locations.CopyFrom(og.GetLocations());

  gu::GpuData<1, float> log_odds(sz);
  log_odds.CopyFrom(og.GetLogOdds());

  // Clear
  data_->input.Clear();

  // Copy over
  int threads = 1024;
  int blocks = std::ceil(static_cast<float>(sz)/threads);
  CopyOccGrid<<<blocks, threads>>>(locations, log_odds, data_->input);
  cudaError_t err = cudaDeviceSynchronize();
  BOOST_ASSERT(err == cudaSuccess);
}

void Network::Apply() {
  data_->cl1.Apply(data_->input, &data_->res_cl1);
  data_->cl2.Apply(data_->res_cl1, &data_->res_cl2);
  data_->cl3.Apply(data_->res_cl2, &data_->res_cl3);
  data_->clatent.Apply(data_->res_cl3, &data_->res_clatent);

  data_->cl_classifier.Apply(data_->res_clatent, &data_->res_classifier);
}

// Load from file
std::vector<float> Network::LoadFile(const fs::path &path) {
  std::vector<float> data;

  std::ifstream file(path.string());

  float val;
  while (file >> val) {
    data.push_back(val);
  }

  return data;
}

std::shared_ptr<Network> Network::LoadNetwork(const fs::path &path) {
  printf("Loading from path: %s\n", path.c_str());

  gu::GpuData<4> l1_weights(7, 7, 14, 200);
  gu::GpuData<1> l1_biases(200);

  gu::GpuData<4> l2_weights(1, 1, 200, 100);
  gu::GpuData<1> l2_biases(100);

  gu::GpuData<4> l3_weights(1, 1, 100, 50);
  gu::GpuData<1> l3_biases(50);

  gu::GpuData<4> latent_weights(1, 1, 50, 25);
  gu::GpuData<1> latent_biases(25);

  gu::GpuData<4> classifier_weights(1, 1, 25, 8);
  gu::GpuData<1> classifier_biases(8);

  l1_weights.CopyFrom(Network::LoadFile(path / "Encoder_l1_weights.dat"));
  l1_biases.CopyFrom(Network::LoadFile(path / "Encoder_l1_biases.dat"));

  l2_weights.CopyFrom(Network::LoadFile(path / "Encoder_l2_weights.dat"));
  l2_biases.CopyFrom(Network::LoadFile(path / "Encoder_l2_biases.dat"));

  l3_weights.CopyFrom(Network::LoadFile(path / "Encoder_l3_weights.dat"));
  l3_biases.CopyFrom(Network::LoadFile(path / "Encoder_l3_biases.dat"));

  latent_weights.CopyFrom(Network::LoadFile(path / "Encoder_latent_weights.dat"));
  latent_biases.CopyFrom(Network::LoadFile(path / "Encoder_latent_biases.dat"));

  classifier_weights.CopyFrom(Network::LoadFile(path / "classifier_weights.dat"));
  classifier_biases.CopyFrom(Network::LoadFile(path / "classifier_biases.dat"));

  ConvolutionalLayer l1(l1_weights, l1_biases);
  ConvolutionalLayer l2(l2_weights, l2_biases);
  ConvolutionalLayer l3(l3_weights, l3_biases);
  ConvolutionalLayer latent(latent_weights, latent_biases);
  ConvolutionalLayer classifier(classifier_weights, classifier_biases);

  printf("Loaded\n");

  NetworkData *data = new NetworkData(l1, l2, l3, latent, classifier);
  return std::shared_ptr<Network>(new Network(data));
}

} // namespace tf
} // namespace library
