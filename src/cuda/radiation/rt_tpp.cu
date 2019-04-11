#include "cuda/constant_mem.h"
#include "cuda/cudaUtility.h"
#include "cuda/kernels.h"
#include "cuda/ptr_util.h"
#include "radiation/spectra.h"
#include "rt_tpp.h"
#include "sim_params.h"
#include "utils/logger.h"
#include "utils/util_functions.h"
#include <curand_kernel.h>

namespace Aperture {

#define MIN_GAMMA 1.01f
#define MAX_GAMMA 1.0e15
#define MIN_LOW_GAMMA 1.0e-5
#define LOW_GAMMA_THR 3.0
#define MIN_EP 1.0e-10
#define MAX_EP 1.0e10

HOST_DEVICE double
total_f(Scalar s) {
  if (s < 4.0) {
    return 0.0;
  } else if (s < 4.6) {
    Scalar s4 = s - 4.0;
    return 0.001 * s4 * s4 *
           (5.6 + 20.4 * s4 - 10.9 * s4 * s4 - 3.6 * s4 * s4 * s4 +
            7.4 * s4 * s4 * s4 * s4);
  } else if (s < 6.0) {
    return 0.582814 - 0.29842 * s + 0.04354 * s * s -
           0.0012977 * s * s * s;
  } else if (s < 14.0) {
    return (3.1247 - 1.3394 * s + 0.14612 * s * s) /
           (1 + 0.4648 * s + 0.016683 * s * s);
  } else {
    Scalar l2s = std::log(2.0 * s);
    return l2s * 28.0 / 9.0 - 218.0 / 27.0 +
           (-(4.0 / 3.0) * l2s * l2s * l2s + 3.863 * l2s * l2s -
            11.0 * l2s + 27.9) /
               s;
  }
}

HOST_DEVICE double
approx_Em(Scalar s) {
  if (s < 100.0) {
    return 0.71 / sqrt(s);
  } else {
    Scalar l2s = std::log(2.0 * s);
    return 0.195 * l2s * l2s / s;
  }
}

HOST_DEVICE double beta(Scalar gamma);

namespace Kernels {

__constant__ Scalar dev_tpp_dg;
__constant__ Scalar* dev_tpp_rate;
__constant__ Scalar* dev_tpp_gammas;
__constant__ Scalar* dev_tpp_Em;

__device__ int
find_tpp_n_gamma(Scalar gamma) {
  if (gamma < MIN_GAMMA) return 0;
  if (gamma > MAX_GAMMA) return dev_params.n_gamma - 1;
  return (std::log(gamma) - std::log(MIN_GAMMA)) / dev_tpp_dg;
}

__device__ Scalar
find_tpp_rate(Scalar gamma) {
  int gn = find_tpp_n_gamma(gamma);
  if (gamma > MAX_GAMMA) return dev_tpp_rate[gn];
  Scalar x = (std::log(gamma) - std::log(dev_tpp_gammas[gn])) / dev_tpp_dg;
  return dev_tpp_rate[gn] * (1.0f - x) + dev_tpp_rate[gn + 1] * x;
}

__device__ Scalar
find_tpp_Em(Scalar gamma) {
  int gn = find_tpp_n_gamma(gamma);
  if (gamma > MAX_GAMMA) return dev_tpp_rate[gn];
  Scalar x = (std::log(gamma) - std::log(dev_tpp_gammas[gn])) / dev_tpp_dg;
  return dev_tpp_Em[gn] * (1.0f - x) + dev_tpp_Em[gn + 1] * x;
}

}  // namespace Kernels

triplet_pairs::triplet_pairs(const SimParams& params)
    : m_rate(params.n_gamma),
      m_Em(params.n_gamma),
      m_gammas(params.n_gamma),
      m_threads(256),
      m_blocks(256),
      m_generator(),
      m_dist(0.0, 1.0) {
  m_dg = (log(MAX_GAMMA) - log(MIN_GAMMA)) /
         ((Scalar)params.n_gamma - 1.0);
  for (uint32_t n = 0; n < m_gammas.size(); n++) {
    m_gammas[n] = exp(log(MIN_GAMMA) + m_dg * (Scalar)n);
  }
  m_gammas.sync_to_device();

  CudaSafeCall(
      cudaMemcpyToSymbol(Kernels::dev_tpp_dg, &m_dg, sizeof(Scalar)));

  Scalar* dev_tpp_rate = m_rate.data_d();
  Scalar* dev_tpp_Em = m_Em.data_d();
  Scalar* dev_g = m_gammas.data_d();
  CudaSafeCall(cudaMemcpyToSymbol(Kernels::dev_tpp_rate, &dev_tpp_rate,
                                  sizeof(Scalar*)));
  CudaSafeCall(cudaMemcpyToSymbol(Kernels::dev_tpp_Em, &dev_tpp_Em,
                                  sizeof(Scalar*)));
  CudaSafeCall(cudaMemcpyToSymbol(Kernels::dev_tpp_gammas, &dev_g,
                                  sizeof(Scalar*)));

  CudaSafeCall(cudaMalloc(&m_states,
                          m_threads * m_blocks * sizeof(curandState)));
  init_rand_states((curandState*)m_states, params.random_seed,
                   m_threads, m_blocks);
}

triplet_pairs::~triplet_pairs() {}

template <typename F>
void
triplet_pairs::init(const F& n_e, Scalar emin, Scalar emax) {
  const int N_mu = 100;
  const int N_e = 800;

  // Compute the gammas and rates for triplet pair production
  Logger::print_info(
      "Pre-calculating the triplet rate and pair mean energy");
  Scalar dmu = 2.0 / (N_mu - 1.0);
  Scalar de = (log(emax) - log(emin)) / (N_e - 1.0);

  for (uint32_t n = 0; n < m_rate.size(); n++) {
    Scalar gamma = m_gammas[n];
    Scalar result = 0.0;
    Scalar Emean = 0.0;
    for (int i_mu = 0; i_mu < N_mu; i_mu++) {
      Scalar mu = i_mu * dmu - 1.0;
      for (int i_e = 0; i_e < N_e; i_e++) {
        Scalar e = exp(log(emin) + i_e * de);
        Scalar b = beta(gamma);
        Scalar s = gamma * e * (1.0 - b * mu);
        result += 0.5f * total_f(s) * (1.0 - b * mu) / 137.0;
        Emean += 0.5f * total_f(s) * approx_Em(s) * (1.0 - b * mu) / 137.0;
        // Scalar sigma = sigma_ic(x);
        // result += 0.5f * n_e(e) * sigma * (1.0f - beta(gamma) * mu) *
        // e;
      }
    }
    m_rate[n] = result * dmu * de;
    m_Em[n] = Emean / result;
  }
  m_rate.sync_to_device();
  m_Em.sync_to_device();
}

template void triplet_pairs::init<Spectra::power_law_hard>(
    const Spectra::power_law_hard& n_e, Scalar emin, Scalar emax);
template void triplet_pairs::init<Spectra::power_law_soft>(
    const Spectra::power_law_soft& n_e, Scalar emin, Scalar emax);
template void triplet_pairs::init<Spectra::black_body>(
    const Spectra::black_body& n_e, Scalar emin, Scalar emax);
template void triplet_pairs::init<Spectra::mono_energetic>(
    const Spectra::mono_energetic& n_e, Scalar emin, Scalar emax);
template void triplet_pairs::init<Spectra::broken_power_law>(
    const Spectra::broken_power_law& n_e, Scalar emin, Scalar emax);

}  // namespace Aperture