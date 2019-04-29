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
  if (s <= 4.01) {
    return 0.0;
  } else if (s <= 100.0) {
    return std::exp((std::log(1.337) + std::log(5.2495 / 1.337) *
                                           std::log(s / 4.01) /
                                           std::log(100.0 / 4.01))) /
           s;
    // } else if (s < 100.0) {
    //   return 0.71 / sqrt(s);
  } else {
    Scalar l2s = std::log(2.0 * s);
    return 0.187 * l2s * l2s / s;
  }
}

HOST_DEVICE Scalar
E_plus_min(Scalar gamma, Scalar s) {
  return (gamma * (s - 1.0f) -
          std::sqrt((gamma * gamma - 1.0f) * s * (s - 4.0f))) /
         (1.0f + 2.0f * s);
}

HOST_DEVICE Scalar
E_plus_max(Scalar gamma, Scalar s) {
  return (gamma * (s - 1.0f) +
          std::sqrt((gamma * gamma - 1.0f) * s * (s - 4.0f))) /
         (1.0f + 2.0f * s);
}

// HOST_DEVICE Scalar
// E_plus_sample(Scalar gamma, Scalar s, float u) {
//   Scalar Emin = E_plus_min(gamma, s);
//   Scalar Emax = E_plus_max(gamma, s);
//   Scalar Emin34 = std::pow(Emin, 0.75f);
//   Scalar Emax34 = std::pow(Emax, 0.75f);
//   return (Emax34 * Emin34) /
//          (std::pow((1.0f - u) / Emin + u / Emax, 1.0f / 3.0f) *
//           (Emax34 * (1.0f - u) + Emin34 * u));
// }

HOST_DEVICE double beta(Scalar gamma);

namespace Kernels {

__constant__ Scalar dev_tpp_dg;
__constant__ Scalar *dev_tpp_rate;
__constant__ Scalar *dev_tpp_gammas;
__constant__ Scalar *dev_tpp_Em;
__constant__ cudaPitchedPtr dev_tpp_dNde;

__device__ int
find_n_gamma(Scalar gamma);

__device__ int
binary_search(float u, Scalar* array, int size, Scalar& l, Scalar& h);

__device__ int
binary_search(float u, int n, cudaPitchedPtr array, Scalar& l,
              Scalar& h);

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
  Scalar x =
      (std::log(gamma) - std::log(dev_tpp_gammas[gn])) / dev_tpp_dg;
  return dev_tpp_rate[gn] * (1.0f - x) + dev_tpp_rate[gn + 1] * x;
}

__device__ Scalar
find_tpp_Em(Scalar gamma) {
  int gn = find_tpp_n_gamma(gamma);
  if (gamma > MAX_GAMMA) return dev_tpp_Em[gn];
  Scalar x =
      (std::log(gamma) - std::log(dev_tpp_gammas[gn])) / dev_tpp_dg;
  return dev_tpp_Em[gn] * (1.0f - x) + dev_tpp_Em[gn + 1] * x;
}

__device__ Scalar
gen_tpp_Ep(Scalar gamma, curandState* state) {
  float u = curand_uniform(state);
  int gn = find_n_gamma(gamma);

  Scalar l, h;
  int b = binary_search(u, gn, dev_tpp_dNde, l, h);
  Scalar bb = (l == h ? b : (u - l) / (h - l) + b);
  Scalar result = std::exp(std::log(MIN_GAMMA) + dev_tpp_dg * bb);

  return min(result, gamma - 1.01);
}

}  // namespace Kernels

triplet_pairs::triplet_pairs(const SimParams &params)
    : m_rate(params.n_gamma),
      m_Em(params.n_gamma),
      m_gammas(params.n_gamma),
      m_dNde(Extent(params.n_gamma, params.n_gamma)),
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

  cudaPitchedPtr p_dNde = m_dNde.data_d().p;
  CudaSafeCall(cudaMemcpyToSymbol(
      Kernels::dev_tpp_dNde, &p_dNde, sizeof(cudaPitchedPtr)));

  Scalar *dev_tpp_rate = m_rate.data_d();
  Scalar *dev_tpp_Em = m_Em.data_d();
  Scalar *dev_g = m_gammas.data_d();
  CudaSafeCall(cudaMemcpyToSymbol(Kernels::dev_tpp_rate, &dev_tpp_rate,
                                  sizeof(Scalar *)));
  CudaSafeCall(cudaMemcpyToSymbol(Kernels::dev_tpp_Em, &dev_tpp_Em,
                                  sizeof(Scalar *)));
  CudaSafeCall(cudaMemcpyToSymbol(Kernels::dev_tpp_gammas, &dev_g,
                                  sizeof(Scalar *)));

  CudaSafeCall(cudaMalloc(&m_states,
                          m_threads * m_blocks * sizeof(curandState)));
  init_rand_states((curandState *)m_states, params.random_seed,
                   m_threads, m_blocks);
}

triplet_pairs::~triplet_pairs() {}

template <typename F>
void
triplet_pairs::init(const F &n_e, Scalar emin, Scalar emax, double n0) {
  const int N_mu = 100;
  const int N_e = 800;

  // Compute the gammas and rates for triplet pair production
  Logger::print_info(
      "Pre-calculating the triplet rate and pair mean energy");
  double dmu = 2.0 / (N_mu - 1.0);
  double de = (log(emax) - log(emin)) / (N_e - 1.0);

  for (uint32_t n = 0; n < m_rate.size(); n++) {
    double gamma = m_gammas[n];
    double result = 0.0;
    double Emean = 0.0;
    for (int i_mu = 0; i_mu < N_mu; i_mu++) {
      double mu = i_mu * dmu - 1.0;
      for (int i_e = 0; i_e < N_e; i_e++) {
        double e = exp(log(emin) + i_e * de);
        double b = beta(gamma);
        double s = gamma * e * (1.0 - b * mu);
        result += 0.5f * n_e(e) * total_f(s) * (1.0 - b * mu) * e;
        Emean += 0.5f * n_e(e) * total_f(s) * approx_Em(s) *
                 (1.0 - b * mu) * e;
        // Scalar sigma = sigma_ic(x);
        // result += 0.5f * n_e(e) * sigma * (1.0f - beta(gamma) * mu) *
        // e;
      }
    }
    m_rate[n] = result * dmu * de * n0 * RE_SQUARE / 137.0;
    // m_Em[n] = Emean / result;
    if (result > 0.0)
      m_Em[n] = Emean / result;
    else
      m_Em[n] = 0.0;
    if (n % 10 == 0) {
      Logger::print_info("TPP rate at gamma {} is {}", gamma,
                         m_rate[n]);
      Logger::print_info("TPP Em at gamma {} is {}", gamma, m_Em[n]);
    }
  }
  m_rate.sync_to_device();
  m_Em.sync_to_device();

  Logger::print_info(
      "Pre-calculating the triplet pair spectrum");
  for (uint32_t n = 0; n < m_gammas.size(); n++) {
    Logger::print_info("Working on gamma {} of {}", n, m_gammas.size());
    double gamma = m_gammas[n];
    for (uint32_t i = 0; i < m_gammas.size(); i++) {
      double E_p = m_gammas[i];
      if (E_p > gamma) {
        m_dNde(i, n) = 0.0;
        continue;
      }
      double result = 0.0;
      // for (uint32_t i_mu = 0; i_mu < N_mu; i_mu++) {
      //   double mu = i_mu * dmu - 1.0;
        for (uint32_t i_e = 0; i_e < N_e; i_e++) {
          double e = exp(log(emin) + i_e * de);
          double ne = n_e(e);
          // if (ne < 1.0e-8) continue;
          // double s = gamma * e * (1.0 - mu);
          double s = gamma * e;
          double Emin = E_plus_min(gamma, s);
          // double Emin = 1.0 / e;
          double Emax = E_plus_max(gamma, s);
          if (E_p < Emin || E_p > Emax) continue;
          // double norm = 4.0/3.0 *
          //               (std::pow(Emin, -0.75) - std::pow(Emax,
          //               -0.75));
          double norm =
              4.0 * (std::pow(Emax, 0.25) - std::pow(Emin, 0.25));
          result +=
              ne * m_rate[n] * m_Em[n] * std::pow(E_p, -1.75) * e / norm;
        }
      // }
      m_dNde(i, n) = result * de * dmu;
    }
    for (uint32_t i = 1; i < m_gammas.size(); i++) {
      m_dNde(i, n) += m_dNde(i - 1, n);
    }
    for (uint32_t i = 0; i < m_gammas.size(); i++) {
      m_dNde(i, n) /= m_dNde(m_gammas.size() - 1, n);
    }
  }
  m_dNde.sync_to_device();
}

template void triplet_pairs::init<Spectra::power_law_hard>(
    const Spectra::power_law_hard &n_e, Scalar emin, Scalar emax,
    double n0);
template void triplet_pairs::init<Spectra::power_law_soft>(
    const Spectra::power_law_soft &n_e, Scalar emin, Scalar emax,
    double n0);
template void triplet_pairs::init<Spectra::black_body>(
    const Spectra::black_body &n_e, Scalar emin, Scalar emax,
    double n0);
template void triplet_pairs::init<Spectra::mono_energetic>(
    const Spectra::mono_energetic &n_e, Scalar emin, Scalar emax,
    double n0);
template void triplet_pairs::init<Spectra::broken_power_law>(
    const Spectra::broken_power_law &n_e, Scalar emin, Scalar emax,
    double n0);

}  // namespace Aperture
