#include "library/flow/solver.cu.h"

#include "library/gpu_util/host_data.cu.h"
#include "library/timer/timer.h"

namespace library {
namespace flow {

Solver::Solver(int nx, int ny, int n_window) :
 nx_(nx),
 ny_(ny),
 n_window_(n_window),
 energy_(nx, ny),
 energy_hat_(nx, ny),
 energy_hat_valid_(nx, ny),
 flow_est_(nx, ny, 2),
 flow_valid_(nx, ny) {
}

__global__ void Expectation(const gu::GpuData<4, float> dist,
                            const gu::GpuData<2, float> p_background,
                            const gu::GpuData<2, int> occ_mask,
                            gu::GpuData<3, int> flow_est,
                            gu::GpuData<2, int> flow_valid,
                            gu::GpuData<2, float> energy,
                            const gu::GpuData<2, float> energy_hat,
                            const gu::GpuData<2, int> energy_hat_valid,
                            int n_window,
                            float w_p) {
  const int bidx = blockIdx.x;
  const int bidy = blockIdx.y;

  const int tidx = threadIdx.x;
  const int tidy = threadIdx.y;

  const int i0 = bidx * blockDim.x;
  const int j0 = bidy * blockDim.y;

  const int i_from = i0 + tidx;
  const int j_from = j0 + tidy;

  // Check in range
  if (!flow_valid.InRange(i_from, j_from)) {
    return;
  }

  if (p_background(i_from, j_from) > 0.5 || occ_mask(i_from, j_from) == 0) {
    flow_est(i_from, j_from, 0) = 0;
    flow_est(i_from, j_from, 1) = 0;

    flow_valid(i_from, j_from) = 0;

    return;
  }

  // Now compute energies
  int best_du = 0;
  int best_dv = 0;
  float best_energy = 0;
  bool valid = false;

  // Smoothing terms
  int sum_u2 = 0;
  int sum_v2 = 0;

  int sum_u = 0;
  int sum_v = 0;

  int count = 0;

  // Init smoothing terms
  int i_n, j_n;
  int u_n, v_n;

  for (int du=-2; du<=2; du++) {
    i_n = i_from + du;
    for (int dv=-2; dv<=2; dv++) {
      if (du==0 && dv==0) {
        continue;
      }

      j_n = j_from + dv;

      if (!flow_valid.InRange(i_n, j_n)) {
        continue;
      }

      if (!flow_valid(i_n, j_n)) {
        continue;
      }

      u_n = flow_est(i_n, j_n, 0);
      v_n = flow_est(i_n, j_n, 1);

      sum_u2 += u_n*u_n;
      sum_v2 += v_n*v_n;

      sum_u += u_n;
      sum_v += v_n;

      count++;
    }
  }

  for (int du=0; du<dist.GetDim(2); du++) {
    for (int dv=0; dv<dist.GetDim(3); dv++) {
      float d = dist(i_from, j_from, du, dv);

      if (d < 0) {
        continue;
      }

      float my_energy = d;

      // Add smoothing
      if (w_p > 0) {
        float smoothness_score = 0.0;

        smoothness_score += sum_u2 + sum_v2;
        smoothness_score += count * (du*du + dv*dv);
        smoothness_score += -2 * (du*sum_u + dv*sum_v);

        my_energy += w_p * smoothness_score;
      }

      // Update best
      if (my_energy < best_energy || !valid) {

        // Is it better than the best score that's already here
        int i_to = i_from + (du - n_window/2);
        int j_to = j_from + (dv - n_window/2);

        int currently_pointing_here = flow_valid(i_from, j_from) &&
                                      flow_est(i_from, j_from, 0) == du &&
                                      flow_est(i_from, j_from, 1) == dv;


        if (!energy_hat_valid(i_to, j_to) ||
            my_energy < energy_hat(i_to, j_to) ||
            currently_pointing_here) {
          valid = true;

          best_du = du;
          best_dv = dv;
          best_energy = my_energy;
        }

      }
    }
  }

  flow_est(i_from, j_from, 0) = best_du;
  flow_est(i_from, j_from, 1) = best_dv;

  flow_valid(i_from, j_from) = valid;

  energy(i_from, j_from) = best_energy;
}

__global__ void Maximization(const gu::GpuData<2, float> energy,
                             gu::GpuData<2, float> energy_hat,
                             gu::GpuData<2, int> energy_hat_valid,
                             const gu::GpuData<3, int> flow_est,
                             gu::GpuData<2, int> flow_valid,
                             int n_window) {
  const int bidx = blockIdx.x;
  const int bidy = blockIdx.y;

  const int tidx = threadIdx.x;
  const int tidy = threadIdx.y;

  const int i0 = bidx * blockDim.x;
  const int j0 = bidy * blockDim.y;

  const int i_dest = i0 + tidx;
  const int j_dest = j0 + tidy;

  // Check in range
  if (!flow_valid.InRange(i_dest, j_dest)) {
    return;
  }

  // Enforce rigid flow
  int best_i_from = 0;
  int best_j_from = 0;
  float best_energy = -1;
  bool valid = false;

  for (int di=0; di<n_window; di++) {
    for (int dj=0; dj<n_window; dj++) {
      int i_from = i_dest - (di - n_window/2);
      int j_from = j_dest - (dj - n_window/2);

      // Check in range
      if (!flow_valid.InRange(i_from, j_from)) {
        continue;
      }

      if (flow_valid(i_from, j_from) &&
          flow_est(i_from, j_from, 0) == di &&
          flow_est(i_from, j_from, 1) == dj &&
          (energy(i_from, j_from) < best_energy || !valid)) {
        best_energy = energy(i_from, j_from);
        best_i_from = i_from;
        best_j_from = j_from;
        valid = true;
      }
    }
  }

  if (!valid) {
    energy_hat_valid(i_dest, j_dest) = 0;
    return;
  }

  // Update best energy
  energy_hat(i_dest, j_dest) = best_energy;
  energy_hat_valid(i_dest, j_dest) = 1;

  // Invalidate other flows
  for (int di=0; di<n_window; di++) {
    for (int dj=0; dj<n_window; dj++) {
      int i_from = i_dest - (di - n_window/2);
      int j_from = j_dest - (dj - n_window/2);

      // Check in range
      if (!flow_valid.InRange(i_from, j_from)) {
        continue;
      }

      // Make sure this isn't the one we choose as best
      if (i_from == best_i_from && j_from == best_j_from) {
        continue;
      }

      // Invalidate it if it leads here
      if (flow_est(i_from, j_from, 0) == di &&
          flow_est(i_from, j_from, 1) == dj) {
        flow_valid(i_from, j_from) = 0;
      }
    }
  }
}

FlowImage Solver::ComputeFlow(const gu::GpuData<4, float> &dist,
                              const gu::GpuData<2, float> &p_background,
                              const gu::GpuData<2, int> &occ_mask,
                              float resolution,
                              int iters) {
  library::timer::Timer timer;

  BOOST_ASSERT(dist.GetDim(0) == nx_);
  BOOST_ASSERT(dist.GetDim(1) == ny_);

  BOOST_ASSERT(p_background.GetDim(0) == nx_);
  BOOST_ASSERT(p_background.GetDim(1) == ny_);

  BOOST_ASSERT(occ_mask.GetDim(0) == nx_);
  BOOST_ASSERT(occ_mask.GetDim(1) == ny_);

  // Setup thread and block dims
  dim3 threads;
  threads.x = 16;
  threads.y = 16;
  threads.z = 1;

  dim3 blocks;
  blocks.x = std::ceil( static_cast<float>(dist.GetDim(0)) / threads.x);
  blocks.y = std::ceil( static_cast<float>(dist.GetDim(1)) / threads.y);
  blocks.z = 1;

  float w_p = kSmoothing_;

  flow_valid_.Clear();
  energy_hat_valid_.Clear();

  timer.Start();
  library::timer::Timer t;
  for (int iter = 0; iter<iters; iter++) {

    t.Start();
    Expectation<<<blocks, threads>>>(dist, p_background, occ_mask, flow_est_, flow_valid_,
        energy_, energy_hat_, energy_hat_valid_, n_window_, iter == 0 ? -1.0:w_p);
    cudaError_t err = cudaDeviceSynchronize();
    BOOST_ASSERT(err == cudaSuccess);
    printf("Expectation took %5.3f ms\n", t.GetMs());

    t.Start();
    Maximization<<<blocks, threads>>>(energy_, energy_hat_,
        energy_hat_valid_, flow_est_, flow_valid_, n_window_);
    err = cudaDeviceSynchronize();
    BOOST_ASSERT(err == cudaSuccess);
    printf("Maximization took %5.3f ms\n", t.GetMs());
  }
  printf("Took %5.3f ms to evaluate EM flow\n", timer.GetMs());

  // Copy from device
  timer.Start();
  gu::HostData<3, int> h_res(flow_est_);
  gu::HostData<2, int> h_valid(flow_valid_);

  FlowImage fi(nx_, ny_, resolution);

  for (int i=0; i<h_res.GetDim(0); i++) {
    for (int j=0; j<h_res.GetDim(1); j++) {
      int ii = fi.MinX() + i;
      int jj = fi.MinY() + j;

      int flow_x = h_res(i, j, 0) - dist.GetDim(2)/2;
      int flow_y = h_res(i, j, 1) - dist.GetDim(3)/2;

      fi.SetFlow(ii, jj, flow_x, flow_y);
      fi.SetFlowValid(ii, jj, h_valid(i, j) == 1);
    }
  }
  printf("Took %5.3f ms to copy from device\n", timer.GetMs());

  return fi;
}

} // namespace tf
} // namespace library
