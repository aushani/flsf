#include "library/tf/network.cu.h"

#include <boost/assert.hpp>
#include <boost/format.hpp>

#include "library/params/params.h"
#include "library/timer/timer.h"

namespace ps = library::params;

namespace library {
namespace tf {

Network::Network(const std::vector<ConvolutionalLayer> &encoder,
                 const ConvolutionalLayer &filter) :
 input_(ps::kOccGridSizeXY, ps::kOccGridSizeXY, ps::kOccGridSizeZ),
 filter_(filter),
 filter_res_(ps::kOccGridSizeXY, ps::kOccGridSizeXY, 2),
 encoder_(encoder) {
  for (const auto &cl : encoder_) {
    intermediate_encoder_results_.emplace_back(ps::kOccGridSizeXY, ps::kOccGridSizeXY, cl.GetOutputLayers());
  }
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

  dense(i, j, k) = val;
}

void Network::SetInput(const rt::OccGrid &og) {
  // Get size
  int sz = og.GetLocations().size();

  // Clear (set unknown)
  input_.Set(0);
  //int threads = 1024;
  //int blocks = std::ceil(static_cast<float>(input_.Size()) / threads);
  //SetUnknown<<<threads, blocks>>>(input_);
  //cudaError_t err = cudaDeviceSynchronize();
  //BOOST_ASSERT(err == cudaSuccess);

  if (og.GetLocations().size() > 0) {
    // Make GpuData objects
    gu::GpuData<1, rt::Location> locations(sz);
    locations.CopyFrom(og.GetLocations());

    gu::GpuData<1, float> log_odds(sz);
    log_odds.CopyFrom(og.GetLogOdds());

    // Copy over
    int threads = 1024;
    int blocks = std::ceil(static_cast<float>(sz)/threads);
    CopyOccGrid<<<blocks, threads>>>(locations, log_odds, input_,
                                    ps::kOccGridMinXY, ps::kOccGridMaxXY,
                                    ps::kOccGridMinXY, ps::kOccGridMaxXY,
                                    ps::kOccGridMinZ, ps::kOccGridMaxZ);
    cudaError_t err = cudaDeviceSynchronize();
    BOOST_ASSERT(err == cudaSuccess);
  }
}

void Network::Apply(gu::GpuData<3, float> *encoding, gu::GpuData<2, float> *p_background,
    gu::GpuData<2, int> *occ_mask) {
  library::timer::Timer t;

  t.Start();
  gu::GpuData<3, float> input = input_;
  for (int layer=0; layer<encoder_.size(); layer++) {
    auto &cl = encoder_[layer];

    if (layer < encoder_.size() - 1) {
      gu::GpuData<3, float> output = intermediate_encoder_results_[layer];
      cl.Apply(input, &output);
      input = output;
    } else {
      cl.Apply(input, encoding);
    }
  }

  filter_.Apply(*encoding, &filter_res_);
  printf("  Took %5.3f ms to run through network\n", t.GetMs());

  t.Start();
  ComputeFilterProbability(p_background);
  ComputeOccMask(occ_mask);
  printf("  Took %5.3f ms to compute data\n", t.GetMs());
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
  blocks.x = std::ceil(static_cast<float>(filter_res_.GetDim(0)) / threads.x);
  blocks.y = std::ceil(static_cast<float>(filter_res_.GetDim(1)) / threads.y);
  blocks.z = 1;

  SoftmaxKernel<<<blocks, threads>>>(filter_res_, *p_background);
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
  return encoder_[encoder_.size() - 1].GetOutputLayers();
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

  std::vector< gu::GpuData<1, float> > biases;
  std::vector< gu::GpuData<4, float> > weights;

  // Figure out dimensions of everything
  size_t i_at = 0;
  while (i_at < dim.size()) {
    biases.emplace_back(dim[i_at]);
    weights.emplace_back(dim[i_at+1], dim[i_at+2], dim[i_at+3], dim[i_at+4]);

    i_at += 5;
  }

  std::vector<ConvolutionalLayer> encoder;
  for (int layer = 0; layer < biases.size() - 1; layer++) {
    auto b = biases[layer];
    auto w = weights[layer];

    fs::path bp;
    fs::path wp;

    if (layer == biases.size() - 2) {
      bp = path / "Encoder_latent_biases.dat";
      wp = path / "Encoder_latent_weights.dat";
    } else {
      bp = path / (boost::format("Encoder_l%d_biases.dat")  % (layer+1)).str();
      wp = path / (boost::format("Encoder_l%d_weights.dat") % (layer+1)).str();
    }

    printf("Loading (%d) from %s\n", b.GetDim(0), bp.c_str());
    printf("Loading (%d, %d, %d, %d) from %s\n", w.GetDim(0), w.GetDim(1), w.GetDim(2), w.GetDim(3), wp.c_str());

    b.CopyFrom(Network::LoadFile(bp));
    w.CopyFrom(Network::LoadFile(wp));

    encoder.emplace_back(ps::kOccGridSizeXY, ps::kOccGridSizeXY, w, b);
  }

  int i_filter = biases.size() - 1;
  auto b = biases[i_filter];
  auto w = weights[i_filter];

  auto bp = path / "Filter_l1_biases.dat";
  auto wp = path / "Filter_l1_weights.dat";

  printf("Loading (%d) from %s\n", b.GetDim(0), bp.c_str());
  printf("Loading (%d, %d, %d, %d) from %s\n", w.GetDim(0), w.GetDim(1), w.GetDim(2), w.GetDim(3), wp.c_str());

  b.CopyFrom(Network::LoadFile(bp));
  w.CopyFrom(Network::LoadFile(wp));

  ConvolutionalLayer filter(ps::kOccGridSizeXY, ps::kOccGridSizeXY, w, b);

  printf("Loaded\n");

  return Network(encoder, filter);
}

} // namespace tf
} // namespace library
