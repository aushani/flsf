#include "library/tf/network.cu.h"

#include <boost/assert.hpp>
#include <boost/format.hpp>

#include "library/gpu_util/host_data.cu.h"
#include "library/params/params.h"
#include "library/timer/timer.h"

namespace ps = library::params;

namespace library {
namespace tf {

Network::Network(const std::vector<ConvolutionalLayer> &encoder,
                 const ConvolutionalLayer &filter) :
 input_(encoder[0].GetInputHeight(), encoder[0].GetInputWidth(), ps::kOccGridSizeZ),
 filter_(filter),
 filter_res_(ps::kOccGridSizeXY, ps::kOccGridSizeXY, 2),
 encoder_(encoder) {
  for (const auto &cl : encoder_) {
    intermediate_encoder_results_.emplace_back(cl.GetOutputHeight(), cl.GetOutputWidth(), cl.GetOutputLayers());
  }
}

__global__ void CopyOccGrid(const gu::GpuData<1, rt::Location> locations, const
    gu::GpuData<1, float> log_odds, gu::GpuData<3, float> dense,
    const int k0, const int k1) {
  // Figure out which hit this thread is processing
  const int bidx = blockIdx.x;
  const int tidx = threadIdx.x;
  const int threads = blockDim.x;

  const int idx = tidx + bidx * threads;
  if (idx >= locations.Size()) {
    return;
  }

  const auto &loc = locations(idx);

  int i = loc.i + (dense.GetDim(0) / 2);
  int j = loc.j + (dense.GetDim(1) / 2);
  //int k = loc.k + (dense.GetDim(2) / 2);
  int k = loc.k - k0;

  if (!dense.InRange(i, j, k)) {
    return;
  }

  float lo = log_odds(idx);
  float p = 1.0 / (1.0 + expf(-lo));

  // Scale to +/- 0.5
  float val = p - 0.5;

  dense(i, j, k) = val;
}

void Network::SetInput(const rt::OccGrid &og) {
  // Get size
  int sz = og.GetLocations().size();

  // Clear (set unknown)
  input_.Set(0.0f);

  if (sz > 0) {
    // Make GpuData objects
    gu::GpuData<1, rt::Location> locations(sz);
    locations.CopyFrom(og.GetLocations());

    gu::GpuData<1, float> log_odds(sz);
    log_odds.CopyFrom(og.GetLogOdds());

    // Copy over
    int threads = 1024;
    int blocks = std::ceil(static_cast<float>(sz)/threads);
    CopyOccGrid<<<blocks, threads>>>(locations, log_odds, input_,
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
    printf("  Layer %d\n", layer);

    auto &cl = encoder_[layer];

    if (layer < encoder_.size() - 1) {
      gu::GpuData<3, float> output = intermediate_encoder_results_[layer];
      cl.Apply(input, &output);
      input = output;
    } else {
      cl.Apply(input, encoding);
    }
  }

  printf("  Classifier\n");
  filter_.Apply(input, &filter_res_);

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

  // Account for huge numbers
  float p_filter = 0.5;
  if (s1 > 20) {
    p_filter = 1.0;
  } else if (s1 < -20) {
    p_filter = 0.0;
  } else {
    //float denom = exp(s1) + exp(s2);
    float denom = exp(s1) + 1;
    p_filter = exp(s1)/denom;
  }

  prob(idx_i, idx_j) = p_filter;
}

void Network::ComputeFilterProbability(gu::GpuData<2, float> *p_background) {
  BOOST_ASSERT(p_background->GetDim(0) == filter_res_.GetDim(0));
  BOOST_ASSERT(p_background->GetDim(1) == filter_res_.GetDim(1));

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

  const int om_i = tidx + bidx * threads_x;
  const int om_j = tidy + bidy * threads_y;

  if (!occ_mask.InRange(om_i, om_j)) {
    return;
  }

  int res = 0;

  int i_star = om_i - (occ_mask.GetDim(0) / 2);
  int j_star = om_j - (occ_mask.GetDim(1) / 2);

  int input_i = i_star + (input.GetDim(0) / 2);
  int input_j = j_star + (input.GetDim(1) / 2);

  for (int k=0; k<input.GetDim(2); k++) {
    if (input(input_i, input_j, k) > 0) {
      res = 1;
      break;
    }
  }

  occ_mask(om_i, om_j) = res;
}

void Network::ComputeOccMask(gu::GpuData<2, int> *occ_mask) {
  BOOST_ASSERT(occ_mask->GetDim(0) <= input_.GetDim(0));
  BOOST_ASSERT(occ_mask->GetDim(1) <= input_.GetDim(1));

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
  /// TODO Cleanup
  printf("Loading from path: %s\n", path.c_str());

  std::vector<int> dim = LoadDimFile(path / "dim.dat");

  std::vector< gu::GpuData<1, float> > biases;
  std::vector< gu::GpuData<4, float> > weights;

  bool zero_padding = false;

  int padding = 0;

  // Figure out dimensions of everything
  size_t i_at = 0;
  while (i_at < dim.size()) {
    biases.emplace_back(dim[i_at]);
    weights.emplace_back(dim[i_at+1], dim[i_at+2], dim[i_at+3], dim[i_at+4]);

    padding += dim[i_at + 2] - 1;

    i_at += 5;
  }

  if (zero_padding) {
    padding = 0;
  } else {
    printf("Padding data by %d total\n", padding);
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

    if (zero_padding) {
      ConvolutionalLayer cl(ps::kOccGridSizeXY, ps::kOccGridSizeXY, w, b, zero_padding);
      cl.SetActivation(Activation::LEAKY_RELU);
      encoder.push_back(cl);
    } else {
      ConvolutionalLayer cl(ps::kOccGridSizeXY + padding, ps::kOccGridSizeXY + padding, w, b, zero_padding);
      cl.SetActivation(Activation::LEAKY_RELU);
      encoder.push_back(cl);

      padding -= w.GetDim(1) - 1;
    }
  }

  // Filter
  int i_filter = biases.size() - 1;
  auto b_filter = biases[i_filter];
  auto w_filter = weights[i_filter];

  auto bp = path / "Filter_l1_biases.dat";
  auto wp = path / "Filter_l1_weights.dat";

  printf("Loading (%d) from %s\n", b_filter.GetDim(0), bp.c_str());
  printf("Loading (%d, %d, %d, %d) from %s\n", w_filter.GetDim(0),
                                               w_filter.GetDim(1),
                                               w_filter.GetDim(2),
                                               w_filter.GetDim(3),
                                               wp.c_str());

  b_filter.CopyFrom(Network::LoadFile(bp));
  w_filter.CopyFrom(Network::LoadFile(wp));

  ConvolutionalLayer filter(ps::kOccGridSizeXY + ((zero_padding) ? 0:padding), ps::kOccGridSizeXY + ((zero_padding) ? 0:padding), w_filter, b_filter, zero_padding);
  filter.SetActivation(Activation::LEAKY_RELU);

  // Add a dummy convolutional layer to get the encoded image to the right dimensions
  // (remember it's a bit padded right now due to the filter classifier)
  int layers = filter.GetInputLayers();
  int height = w_filter.GetDim(1);
  int width = w_filter.GetDim(2);
  gu::HostData<4, float> w_dummy(layers, height, width, layers);
  gu::HostData<1, float> b_dummy(layers);

  w_dummy.Set(0.0f);
  for (int i=0; i<layers; i++) {
    w_dummy(i, height/2, width/2, i) = 1.0;
  }

  b_dummy.Set(0.0f);

  encoder.emplace_back(ps::kOccGridSizeXY + ((zero_padding) ? 0:padding),
                       ps::kOccGridSizeXY + ((zero_padding) ? 0:padding),
                       gu::GpuData<4, float>(w_dummy),
                       gu::GpuData<1, float>(b_dummy),
                       zero_padding);

  // no activation

  printf("Loaded\n");

  return Network(encoder, filter);
}

} // namespace tf
} // namespace library
