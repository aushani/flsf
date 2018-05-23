#include "library/tf/network.cu.h"

#include <boost/assert.hpp>

#include "library/params/params.h"

namespace ps = library::params;

namespace library {
namespace tf {

Network::Network(const ConvolutionalLayer &l1,
                 const ConvolutionalLayer &l2,
                 const ConvolutionalLayer &l3,
                 const ConvolutionalLayer &latent,
                 const ConvolutionalLayer &cl_filter) :
 cl1_(l1),
 cl2_(l2),
 cl3_(l3),
 clatent_(latent),
 cl_filter_(cl_filter),
 input_(167, 167, 13),
 res_cl1_(167, 167, l1.GetOutputLayers()),
 res_cl2_(167, 167, l2.GetOutputLayers()),
 res_cl3_(167, 167, l3.GetOutputLayers()),
 res_filter_(167, 167, cl_filter.GetOutputLayers()) {
  input_.SetCoalesceDim(0);
  res_cl1_.SetCoalesceDim(0);
  res_cl2_.SetCoalesceDim(0);
  res_cl3_.SetCoalesceDim(0);
  res_filter_.SetCoalesceDim(0);
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
    gu::GpuData<1, float> log_odds, gu::GpuData<3, float> dense,
    const int i0, const int i1, const int j0, const int j1, const int k0, const int k1) {
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

  int i = loc.i;
  int j = loc.j;
  int k = loc.k;

  if (i < i0 || i >= i1) {
    return;
  }

  if (j < j0 || j >= j1) {
    return;
  }

  if (k < k0 || k >= k1) {
    return;
  }

  i -= i0;
  j -= j0;
  k -= k0;

  float val = p - 0.5;
  //float val = p;

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
    CopyOccGrid<<<blocks, threads>>>(locations, log_odds, input_,
                                    ps::kOccGridMinXY, ps::kOccGridMaxXY,
                                    ps::kOccGridMinXY, ps::kOccGridMaxXY,
                                    ps::kOccGridMinZ, ps::kOccGridMaxZ);
    err = cudaDeviceSynchronize();
    BOOST_ASSERT(err == cudaSuccess);
  }
}

void Network::Apply(gu::GpuData<3, float> *encoding, gu::GpuData<2, float> *p_background,
    gu::GpuData<2, int> *occ_mask) {
  cl1_.Apply(input_, &res_cl1_);
  cl2_.Apply(res_cl1_, &res_cl2_);
  cl3_.Apply(res_cl2_, &res_cl3_);
  clatent_.Apply(res_cl3_, encoding);

  cl_filter_.Apply(*encoding, &res_filter_);

  ComputeFilterProbability(p_background);
  ComputeOccMask(occ_mask);
}

__global__ void SoftmaxKernel(const gu::GpuData<3, float> res, gu::GpuData<2, float> prob) {
  // Figure out which i, j this thread is processing
  const int bidx = blockIdx.x;
  const int bidy = blockIdx.y;

  const int tidx = threadIdx.x;
  const int tidy = threadIdx.y;

  const int threads_x = blockDim.x;
  const int threads_y = blockDim.y;

  const int idx_i = tidx + bidx * threads_x;
  const int idx_j = tidy + bidy * threads_y;

  if (!prob.InRange(idx_i, idx_j)) {
    return;
  }

  float s1 = res(idx_i, idx_j, 0);
  float s2 = res(idx_i, idx_j, 1);

  // For scaling
  s1 -= s2;
  s2 -= s2;

  //float denom = exp(s1) + exp(s2);
  float denom = exp(s1) + 1;

  float p_filter = exp(s1)/denom;

  prob(idx_i, idx_j) = p_filter;
}

void Network::ComputeFilterProbability(gu::GpuData<2, float> *p_background) {
  dim3 threads;
  threads.x = 32;
  threads.y = 32;
  threads.z = 1;

  dim3 blocks;
  blocks.x = std::ceil(static_cast<float>(res_filter_.GetDim(0)) / threads.x);
  blocks.y = std::ceil(static_cast<float>(res_filter_.GetDim(1)) / threads.y);
  blocks.z = 1;

  SoftmaxKernel<<<blocks, threads>>>(res_filter_, *p_background);
  cudaError_t err = cudaDeviceSynchronize();
  BOOST_ASSERT(err == cudaSuccess);
}

__global__ void GetOccMask(const gu::GpuData<3, float> input, gu::GpuData<2, int> occ_mask) {
  // Figure out which i, j this thread is processing
  const int bidx = blockIdx.x;
  const int bidy = blockIdx.y;

  const int tidx = threadIdx.x;
  const int tidy = threadIdx.y;

  const int threads_x = blockDim.x;
  const int threads_y = blockDim.y;

  const int idx_i = tidx + bidx * threads_x;
  const int idx_j = tidy + bidy * threads_y;

  if (!occ_mask.InRange(idx_i, idx_j)) {
    return;
  }

  int res = 0;

  for (int k=0; k<input.GetDim(2); k++) {
    if (input(idx_i, idx_j, k) > 0) {
      res = 1;
      break;
    }
  }

  occ_mask(idx_i, idx_j) = res;
}

void Network::ComputeOccMask(gu::GpuData<2, int> *occ_mask) {
  BOOST_ASSERT(occ_mask->GetDim(0) == input_.GetDim(0));
  BOOST_ASSERT(occ_mask->GetDim(1) == input_.GetDim(1));

  dim3 threads;
  threads.x = 32;
  threads.y = 32;
  threads.z = 1;

  dim3 blocks;
  blocks.x = std::ceil(static_cast<float>(occ_mask->GetDim(0)) / threads.x);
  blocks.y = std::ceil(static_cast<float>(occ_mask->GetDim(1)) / threads.y);
  blocks.z = 1;

  GetOccMask<<<blocks, threads>>>(input_, *occ_mask);
  cudaError_t err = cudaDeviceSynchronize();
  BOOST_ASSERT(err == cudaSuccess);
}

int Network::GetEncodingDim() const {
  return clatent_.GetOutputLayers();
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

std::vector<int> Network::LoadDimFile(const fs::path &path) {
  std::vector<int> data;

  std::ifstream file(path.string());

  int val;
  while (file >> val) {
    data.push_back(val);
  }

  return data;
}

Network Network::LoadNetwork(const fs::path &path) {
  printf("Loading from path: %s\n", path.c_str());

  std::vector<int> dim = LoadDimFile(path / "dim.dat");

  gu::GpuData<1, float> l1_biases(dim[0]);
  gu::GpuData<4, float> l1_weights(dim[1], dim[2], dim[3], dim[4]);

  gu::GpuData<1, float> l2_biases(dim[5]);
  gu::GpuData<4, float> l2_weights(dim[6], dim[7], dim[8], dim[9]);

  gu::GpuData<1, float> l3_biases(dim[10]);
  gu::GpuData<4, float> l3_weights(dim[11], dim[12], dim[13], dim[14]);

  gu::GpuData<1, float> latent_biases(dim[15]);
  gu::GpuData<4, float> latent_weights(dim[16], dim[17], dim[18], dim[19]);

  gu::GpuData<1, float> filter_biases(dim[20]);
  gu::GpuData<4, float> filter_weights(dim[21], dim[22], dim[23], dim[24]);

  l1_weights.CopyFrom(Network::LoadFile(path / "Encoder_l1_weights.dat"));
  l1_biases.CopyFrom(Network::LoadFile(path / "Encoder_l1_biases.dat"));

  l2_weights.CopyFrom(Network::LoadFile(path / "Encoder_l2_weights.dat"));
  l2_biases.CopyFrom(Network::LoadFile(path / "Encoder_l2_biases.dat"));

  l3_weights.CopyFrom(Network::LoadFile(path / "Encoder_l3_weights.dat"));
  l3_biases.CopyFrom(Network::LoadFile(path / "Encoder_l3_biases.dat"));

  latent_weights.CopyFrom(Network::LoadFile(path / "Encoder_latent_weights.dat"));
  latent_biases.CopyFrom(Network::LoadFile(path / "Encoder_latent_biases.dat"));

  //classifier_weights.CopyFrom(Network::LoadFile(path / "classifier_weights.dat"));
  //classifier_biases.CopyFrom(Network::LoadFile(path / "classifier_biases.dat"));

  filter_weights.CopyFrom(Network::LoadFile(path / "Filter_l1_weights.dat"));
  filter_biases.CopyFrom(Network::LoadFile(path / "Filter_l1_biases.dat"));

  ConvolutionalLayer l1(l1_weights, l1_biases);
  ConvolutionalLayer l2(l2_weights, l2_biases);
  ConvolutionalLayer l3(l3_weights, l3_biases);
  ConvolutionalLayer latent(latent_weights, latent_biases);
  //ConvolutionalLayer classifier(classifier_weights, classifier_biases);
  ConvolutionalLayer filter(filter_weights, filter_biases);

  printf("Loaded\n");

  //return Network(l1, l2, l3, latent, classifier);
  return Network(l1, l2, l3, latent, filter);
}

} // namespace tf
} // namespace library
