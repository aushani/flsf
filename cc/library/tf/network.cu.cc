#include "library/tf/network.cu.h"

#include <boost/assert.hpp>

namespace library {
namespace tf {

Network::Network(const ConvolutionalLayer &l1, const ConvolutionalLayer &l2, const ConvolutionalLayer &l3, const ConvolutionalLayer &latent, const ConvolutionalLayer &clc) :
 cl1_(l1),
 cl2_(l2),
 cl3_(l3),
 clatent_(latent),
 cl_classifier_(clc),
 input_(167, 167, 12),
 res_cl1_(167, 167, 200),
 res_cl2_(167, 167, 100),
 res_cl3_(167, 167, 50),
 res_clatent_(167, 167, 25),
 res_classifier_(167, 167, 8) {
  input_.SetCoalesceDim(0);
  res_cl1_.SetCoalesceDim(0);
  res_cl2_.SetCoalesceDim(0);
  res_cl3_.SetCoalesceDim(0);
  res_clatent_.SetCoalesceDim(0);
  res_classifier_.SetCoalesceDim(0);
}

__global__ void SetUnknown(gu::GpuData<3, float> dense) {
  // Figure out which hit this thread is processing
  const int bidx = blockIdx.x;
  const int tidx = threadIdx.x;
  const int threads = blockDim.x;

  const int idx = tidx + bidx * threads;
  if (idx >= dense.Size()) {
    return;
  }

  //dense.GetRawPointer()[idx] = 0.5;
  dense.GetRawPointer()[idx] = 0.0;
}

__global__ void CopyOccGrid(const gu::GpuData<1, rt::Location> locations, const
    gu::GpuData<1, float> log_odds, gu::GpuData<3, float> dense) {
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

  int i = loc.i + dense.GetDim(0)/2;
  int j = loc.j + dense.GetDim(1)/2;
  int k = loc.k + dense.GetDim(2)/2;

  if (i < 0 || i >= dense.GetDim(0)) {
    return;
  }

  if (j < 0 || j >= dense.GetDim(1)) {
    return;
  }

  if (k < 0 || k >= dense.GetDim(2)) {
    return;
  }

  float val = p - 0.5;

  dense(i, j, k) = val;
}

void Network::SetInput(const rt::OccGrid &og) {
  // Get size
  int sz = og.GetLocations().size();

  // Clear (set unknown)
  int threads = 1024;
  int blocks = std::ceil(static_cast<float>(input_.Size()) / threads);
  SetUnknown<<<threads, blocks>>>(input_);
  cudaError_t err = cudaDeviceSynchronize();
  BOOST_ASSERT(err == cudaSuccess);

  if (og.GetLocations().size() > 0) {
    // Make GpuData objects
    gu::GpuData<1, rt::Location> locations(sz);
    locations.CopyFrom(og.GetLocations());

    gu::GpuData<1, float> log_odds(sz);
    log_odds.CopyFrom(og.GetLogOdds());

    // Copy over
    threads = 1024;
    blocks = std::ceil(static_cast<float>(sz)/threads);
    CopyOccGrid<<<blocks, threads>>>(locations, log_odds, input_);
    err = cudaDeviceSynchronize();
    BOOST_ASSERT(err == cudaSuccess);
  }
}

const gu::GpuData<3, float>& Network::GetEncoding() const {
  return res_clatent_;
}

const gu::GpuData<3, float>& Network::GetClassification() const {
  return res_classifier_;
}

void Network::Apply() {
  cl1_.Apply(input_, &res_cl1_);
  cl2_.Apply(res_cl1_, &res_cl2_);
  cl3_.Apply(res_cl2_, &res_cl3_);
  clatent_.Apply(res_cl3_, &res_clatent_);

  cl_classifier_.Apply(res_clatent_, &res_classifier_);
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

Network Network::LoadNetwork(const fs::path &path) {
  printf("Loading from path: %s\n", path.c_str());

  gu::GpuData<4, float> l1_weights(7, 7, 12, 200);
  gu::GpuData<1, float> l1_biases(200);

  gu::GpuData<4, float> l2_weights(1, 1, 200, 100);
  gu::GpuData<1, float> l2_biases(100);

  gu::GpuData<4, float> l3_weights(1, 1, 100, 50);
  gu::GpuData<1, float> l3_biases(50);

  gu::GpuData<4, float> latent_weights(1, 1, 50, 25);
  gu::GpuData<1, float> latent_biases(25);

  gu::GpuData<4, float> classifier_weights(1, 1, 25, 8);
  gu::GpuData<1, float> classifier_biases(8);

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

  return Network(l1, l2, l3, latent, classifier);
}

} // namespace tf
} // namespace library
