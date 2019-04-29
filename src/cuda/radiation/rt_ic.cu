#include "cuda/constant_mem.h"
#include "cuda/cudaUtility.h"
#include "cuda/kernels.h"
#include "cuda/ptr_util.h"
#include "cuda/radiation/rt_ic.h"
#include "radiation/spectra.h"
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
beta(Scalar gamma) {
  return 1.0 / sqrt(1.0 - 1.0 / square(gamma));
}

HOST_DEVICE double
sigma_ic(Scalar x) {
  if (x < 1.0e-3) {
    return 1.0f - 2.0f * x + 26.0f * x * x / 5.0f;
  } else {
    double l = std::log(1.0 + 2.0 * x);
    return 0.75 *
           ((1.0 + x) * (2.0 * x * (1.0 + x) / (1.0 + 2.0 * x) - l) /
                cube(x) +
            0.5 * l / x - (1.0 + 3.0 * x) / square(1.0 + 2.0 * x));
  }
}

HOST_DEVICE double
sigma_gg(double beta) {
  return (1.0 - square(beta)) * ((3.0 - beta * beta * beta * beta) *
                                     log((1.0 + beta) / (1.0 - beta)) -
                                 2.0 * beta * (2.0 - beta * beta));
}

HOST_DEVICE double
x_ic(Scalar gamma, Scalar e, Scalar mu) {
  return gamma * e * (1.0 - mu * beta(gamma));
}

HOST_DEVICE double
sigma_lab(Scalar q, Scalar ge) {
  return 2.0 * q * log(q) + (1.0 + 2.0 * q) * (1.0 - q) +
         0.5 * square(ge * q) * (1.0 - q) / (1.0 + q * ge);
}

HOST_DEVICE double
sigma_rest(Scalar ep, Scalar e1p) {
  return (ep / e1p + e1p / ep -
          (1.0 - square(1.0 - 1.0 / e1p + 1.0 / ep)));
}

namespace Kernels {

__constant__ Scalar dev_ic_dep;
__constant__ Scalar dev_ic_dg;
__constant__ Scalar dev_ic_dlep;
__constant__ cudaPitchedPtr dev_ic_dNde;
__constant__ cudaPitchedPtr dev_ic_dNde_thompson;
__constant__ Scalar* dev_ic_rate;
__constant__ Scalar* dev_gg_rate;
__constant__ Scalar* dev_gammas;

__device__ int
find_n_gamma(Scalar gamma) {
  if (gamma < MIN_GAMMA) return 0;
  if (gamma > MAX_GAMMA) return dev_params.n_gamma - 1;
  return (std::log(gamma) - std::log(MIN_GAMMA)) / dev_ic_dg;
}

__device__ int
find_n_e1(Scalar e1) {
  if (e1 < 0.0f) e1 = 0.0;
  if (e1 > 1.0f) e1 = 1.0f;
  return e1 / dev_ic_dep;
}

__device__ int
binary_search(float u, Scalar* array, int size, Scalar& l, Scalar& h) {
  int a = 0, b = size - 1, tmp = 0;
  Scalar v = 0.0f;
  while (a < b) {
    tmp = (a + b) / 2;
    v = array[tmp];
    if (v < u) {
      a = tmp + 1;
    } else if (v > u) {
      b = tmp - 1;
    } else {
      b = tmp;
      l = h = v;
      return b;
    }
  }
  if (v < u) {
    l = v;
    h = (a < size ? array[a] : v);
    return tmp;
  } else {
    h = v;
    l = (b >= 0 ? array[b] : v);
    return max(b, 0);
  }
}

__device__ int
binary_search(float u, int n, cudaPitchedPtr array, Scalar& l,
              Scalar& h) {
  int size = array.xsize / sizeof(Scalar);
  Scalar* ptr = (Scalar*)((char*)array.ptr + n * array.pitch);
  return binary_search(u, ptr, size, l, h);
}

__device__ Scalar
gen_photon_e(Scalar gamma, curandState* state) {
  float u = curand_uniform(state);
  Scalar bb;
  double result;
  int gn = find_n_gamma(gamma);
  // if (gamma < LOW_GAMMA_THR) {
  //   // Thompson regime
  //   Scalar l, h;
  //   int b = binary_search(u, gn, dev_ic_dNde_thompson, l, h);
  //   bb = (l == h ? b : (u - l) / (h - l) + b);
  //   result = exp(dev_ic_dlep * bb + log(MIN_EP));
  // } else {
  Scalar l, h;
  int b = binary_search(u, gn, dev_ic_dNde, l, h);
  if (b < 2) {
    b = binary_search(u, gn, dev_ic_dNde_thompson, l, h);
    bb = (l == h ? b : (u - l) / (h - l) + b);
    result = exp(dev_ic_dlep * bb + log(MIN_EP));
  } else {
    bb = (l == h ? b : (u - l) / (h - l) + b);
    result = dev_ic_dep * bb;
  }

  return min(result * gamma, gamma - 1.01);
}

__device__ Scalar
find_ic_rate(Scalar gamma) {
  int gn = find_n_gamma(gamma);
  if (gamma > MAX_GAMMA) return dev_ic_rate[gn];
  Scalar x = (std::log(gamma) - std::log(dev_gammas[gn])) / dev_ic_dg;
  return dev_ic_rate[gn] * (1.0f - x) + dev_ic_rate[gn + 1] * x;
}

__device__ Scalar
find_gg_rate(Scalar eph) {
  if (eph < 2.0) return 0.0;
  int n = find_n_gamma(eph);
  if (eph > MAX_GAMMA) return dev_gg_rate[n];
  Scalar x = (std::log(eph) - std::log(dev_gammas[n])) / dev_ic_dg;
  return dev_gg_rate[n] * (1.0f - x) + dev_gg_rate[n + 1] * x;
}

template <typename F>
__global__ void
init_scatter_rate(F n_e, Scalar dmu, Scalar de, int N_mu, int N_e,
                  Scalar* rate, Scalar* gammas, Scalar emin,
                  uint32_t ngamma) {
  Scalar result = 0.0;
  for (int n = threadIdx.x + blockIdx.x * blockDim.x; n < ngamma;
       n += blockDim.x * gridDim.x) {
    printf("%d\n", n);
    Scalar gamma = gammas[n];

    for (int i_mu = 0; i_mu < N_mu; i_mu++) {
      Scalar mu = i_mu * dmu - 1.0;
      for (int i_e = 0; i_e < N_e; i_e++) {
        Scalar e = exp(log(emin) + i_e * de);
        Scalar x = x_ic(gamma, e, mu);
        Scalar sigma = sigma_ic(x);
        result += 0.5f * n_e(e) * sigma * (1.0f - beta(gamma) * mu) * e;
      }
    }

    rate[n] = result;
  }
}

__global__ void
gen_photon_energies(Scalar* eph, Scalar* gammas, uint32_t num,
                    curandState* states) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  curandState local_state = states[id];
  for (int n = threadIdx.x + blockIdx.x * blockDim.x; n < num;
       n += blockDim.x * gridDim.x) {
    float u = curand_uniform(&local_state);
    if (u < find_ic_rate(gammas[n]))
      eph[n] = gen_photon_e(gammas[n], &local_state);
    else
      eph[n] = 0.0;
  }
  states[id] = local_state;
}

__global__ void
gen_rand_gammas(Scalar* gammas, uint32_t num, curandState* states) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  curandState local_state = states[id];
  for (int n = threadIdx.x + blockIdx.x * blockDim.x; n < num;
       n += blockDim.x * gridDim.x) {
    gammas[n] = std::exp(std::log(1.01) +
                         curand_uniform(&local_state) *
                             (std::log(1e10) - std::log(1.01)));
  }
  states[id] = local_state;
}

}  // namespace Kernels

inverse_compton::inverse_compton(const SimParams& params)
    : m_dNde(Extent(params.n_ep, params.n_gamma)),
      m_dNde_thompson(Extent(params.n_ep, params.n_gamma)),
      m_ic_rate(params.n_gamma),
      m_gg_rate(params.n_gamma),
      m_gammas(params.n_gamma),
      m_log_ep(params.n_ep),
      m_ep(params.n_ep),
      m_threads(256),
      m_blocks(256),
      m_generator(),
      m_dist(0.0, 1.0) {
  m_dep = 1.0 / ((Scalar)m_ep.size() - 1.0);
  for (uint32_t n = 0; n < m_ep.size(); n++) {
    m_ep[n] = m_dep * n;
  }
  m_dlep = (-log(MIN_EP)) / ((Scalar)m_ep.size() - 1.0);
  for (uint32_t n = 0; n < m_ep.size(); n++) {
    m_log_ep[n] = exp(m_dlep * n + log(MIN_EP));
    // Logger::print_info("log ep is {}", m_log_ep[n]);
  }
  m_dg = (log(MAX_GAMMA) - log(MIN_GAMMA)) /
         ((Scalar)params.n_gamma - 1.0);
  for (uint32_t n = 0; n < m_gammas.size(); n++) {
    m_gammas[n] = exp(log(MIN_GAMMA) + m_dg * (Scalar)n);
  }

  m_ep.sync_to_device();
  m_gammas.sync_to_device();
  m_log_ep.sync_to_device();

  CudaSafeCall(
      cudaMemcpyToSymbol(Kernels::dev_ic_dep, &m_dep, sizeof(Scalar)));
  CudaSafeCall(
      cudaMemcpyToSymbol(Kernels::dev_ic_dg, &m_dg, sizeof(Scalar)));
  CudaSafeCall(cudaMemcpyToSymbol(Kernels::dev_ic_dlep, &m_dlep,
                                  sizeof(Scalar)));
  cudaPitchedPtr p_dNde =m_dNde.data_d().p;
  cudaPitchedPtr p_dNde_t =m_dNde_thompson.data_d().p;

  CudaSafeCall(cudaMemcpyToSymbol(
      Kernels::dev_ic_dNde, &p_dNde, sizeof(cudaPitchedPtr)));
  CudaSafeCall(cudaMemcpyToSymbol(Kernels::dev_ic_dNde_thompson,
                                  &p_dNde_t,
                                  sizeof(cudaPitchedPtr)));
  Scalar* dev_ic_rates = m_ic_rate.data_d();
  Scalar* dev_gg_rates = m_gg_rate.data_d();
  Scalar* dev_g = m_gammas.data_d();
  CudaSafeCall(cudaMemcpyToSymbol(Kernels::dev_ic_rate, &dev_ic_rates,
                                  sizeof(Scalar*)));
  CudaSafeCall(cudaMemcpyToSymbol(Kernels::dev_gg_rate, &dev_gg_rates,
                                  sizeof(Scalar*)));
  CudaSafeCall(
      cudaMemcpyToSymbol(Kernels::dev_gammas, &dev_g, sizeof(Scalar*)));

  CudaSafeCall(cudaMalloc(&m_states,
                          m_threads * m_blocks * sizeof(curandState)));
  init_rand_states((curandState*)m_states, params.random_seed,
                   m_threads, m_blocks);
}

inverse_compton::~inverse_compton() {}

template <typename F>
void
inverse_compton::init(const F& n_e, Scalar emin, Scalar emax,
                      double n0) {
  const int N_mu = 100;
  const int N_e = 800;

  // Compute the gammas and rates for IC scattering
  Logger::print_info("Pre-calculating the scattering rate");
  double dmu = 2.0 / (N_mu - 1.0);
  double de = (log(emax) - log(emin)) / (N_e - 1.0);
  // Kernels::init_scatter_rate<<<200, 200>>>(
  //     n_e, dmu, de, N_mu, N_e, m_ic_rate.data(), m_gammas.data(),
  //     emin, m_gammas.size());
  // cudaDeviceSynchronize();
  // CudaCheckError();
  // m_ic_rate.sync_to_host();
  for (uint32_t n = 0; n < m_ic_rate.size(); n++) {
    double gamma = m_gammas[n];
    // Logger::print_info("gamma is {}", gamma);
    double result = 0.0;
    for (int i_mu = 0; i_mu < N_mu; i_mu++) {
      double mu = i_mu * dmu - 1.0;
      for (int i_e = 0; i_e < N_e; i_e++) {
        double e = exp(log(emin) + i_e * de);
        double x = x_ic(gamma, e, mu);
        double sigma = sigma_ic(x);
        result += 0.5f * n_e(e) * sigma * (1.0f - beta(gamma) * mu) * e;
      }
    }
    m_ic_rate[n] =
        result * dmu * de * (RE_SQUARE * n0) * 8.0 * CONST_PI / 3.0;
    if (n % 10 == 0)
      Logger::print_info("IC rate at gamma {} is {}, result is {}, factor is {}", gamma,
                         m_ic_rate[n], result, (dmu * de * RE_SQUARE * n0) * 8.0 * CONST_PI / 3.0);

    // if (n != 0)
    //   m_ic_rate[n] /= m_ic_rate[0];
    // Logger::print_info("IC rate is {}", m_ic_rate[n]);
  }
  // m_ic_rate[0] = 1.0;
  m_ic_rate.sync_to_device();

  // TODO: Precalculate gamma-gamma rate
  Logger::print_info(
      "Pre-calculating the gamma-gamma pair creation rate");
  for (uint32_t n = 0; n < m_gg_rate.size(); n++) {
    double eph = m_gammas[n];
    if (eph < 2.0) {
      m_gg_rate[n] = 0.0;
    } else {
      double result = 0.0;
      for (int i_mu = 0; i_mu < N_mu; i_mu++) {
        double mu = i_mu * dmu - 1.0;
        for (int i_e = 0; i_e < N_e; i_e++) {
          double e = exp(log(emin) + i_e * de);
          double s = eph * e * (1.0 - mu) * 0.5;
          if (s <= 1.0) continue;
          double b = sqrt(1.0 - 1.0 / s);
          if (b == 1.0) continue;
          result += sigma_gg(b) * (1.0 - mu) * n_e(e) * e;
          // Logger::print_info("eph is {}, s is {}, b is {}, sigma is
          // {}", eph, s, b, sigma_gg(b));
        }
      }
      m_gg_rate[n] =
          0.25 * result * dmu * de * CONST_PI * (n0 * RE_SQUARE);
      // if (n != 0)
      //   m_gg_rate[n] /= m_gg_rate[0];
      if (n % 10 == 0)
        Logger::print_info("gg rate at gamma {} is {}", eph,
                           m_gg_rate[n]);
    }
  }
  // m_gg_rate[0] = 1.0;
  m_gg_rate.sync_to_device();

  Logger::print_info("Pre-calculating the lab-frame spectrum");
  for (uint32_t n = 0; n < m_gammas.size(); n++) {
    double gamma = m_gammas[n];
    for (uint32_t i = 0; i < m_ep.size(); i++) {
      double e1 = m_ep[i];
      double result = 0.0;
      for (uint32_t i_e = 0; i_e < N_e; i_e++) {
        double e = exp(log(emin) + i_e * de);
        double ne = n_e(e);
        // if (ne < 1.0e-8) continue;
        double ge = gamma * e * 4.0;
        double q = e1 / (ge * (1.0 - e1));
        if (e1 < ge / (1.0 + ge) && e1 > e / gamma)
          // result += n_e(e) * sigma_lab(q, ge) / gamma;
          result += ne * sigma_lab(q, ge) / gamma;
      }
      m_dNde(i, n) = result * de;
    }
    for (uint32_t i = 1; i < m_ep.size(); i++) {
      m_dNde(i, n) += m_dNde(i - 1, n);
    }
    for (uint32_t i = 0; i < m_ep.size(); i++) {
      m_dNde(i, n) /= m_dNde(m_ep.size() - 1, n);
    }
  }

  m_dNde.sync_to_device();

  for (uint32_t n = 0; n < m_gammas.size(); n++) {
    double gamma = m_gammas[n];
    for (uint32_t i = 0; i < m_log_ep.size(); i++) {
      double e1 = m_log_ep[i];
      double result = 0.0;
      for (uint32_t i_e = 0; i_e < N_e; i_e++) {
        double e = exp(log(emin) + i_e * de);
        double ne = n_e(e);
        // if (ne < 1.0e-8) continue;
        double ge = gamma * e * 4.0;
        double q = e1 / (ge * (1.0 - e1));
        if (e1 < ge / (1.0 + ge) && e1 > e / gamma)
          result += ne * sigma_lab(q, ge) / gamma;
      }
      m_dNde_thompson(i, n) = result * de * e1;
    }
    for (uint32_t i = 1; i < m_ep.size(); i++) {
      m_dNde_thompson(i, n) += m_dNde_thompson(i - 1, n);
    }
    for (uint32_t i = 0; i < m_ep.size(); i++) {
      m_dNde_thompson(i, n) /= m_dNde_thompson(m_ep.size() - 1, n);
    }
  }

  m_dNde_thompson.sync_to_device();
  // m_npep.sync_to_device();
  // m_dnde1p.sync_to_device();
}

int
inverse_compton::find_n_gamma(Scalar gamma) const {
  if (gamma < MIN_GAMMA) return 0;
  if (gamma > MAX_GAMMA) return m_gammas.size() - 1;
  return (std::log(gamma) - std::log(MIN_GAMMA)) / m_dg;
}

// int
// inverse_compton::find_n_low_gamma(Scalar gamma) const {
//   if (gamma < MIN_LOW_GAMMA) return 0;
//   if (gamma > LOW_GAMMA_THR) return m_low_gammas.size() - 1;
//   return (std::log(gamma) - std::log(MIN_LOW_GAMMA)) / m_dgl;
// }

int
inverse_compton::binary_search(float u, int n,
                               const cu_multi_array<Scalar>& array,
                               Scalar& v1, Scalar& v2) const {
  int a = 0, b = m_ep.size() - 1, tmp = 0;
  Scalar v = 0.0f;
  while (a < b) {
    tmp = (a + b) / 2;
    v = array(tmp, n);
    if (v < u) {
      a = tmp + 1;
    } else if (v > u) {
      b = tmp - 1;
    } else {
      b = tmp;
      v1 = v2 = v;
      return b;
    }
  }
  if (v < u) {
    v1 = v;
    v2 = (a < m_ep.size() ? array(a, n) : v);
    return tmp;
  } else {
    v2 = v;
    v1 = (b >= 0 ? array(b, n) : v);
    return std::max(b, 0);
  }
}

Scalar
inverse_compton::gen_photon_e(Scalar gamma) {
  // int n = find_n_gamma(gamma);
  // Scalar e1p = gen_e1p(n);
  // Scalar ep = gen_ep(n, e1p);
  // if (ep < e1p) ep = e1p;
  // Scalar mu = 1.0 - 1.0 / e1p + 1.0 / ep;
  // // Scalar result = gamma * e1p * (1.0 + beta(gamma) * (-mu));
  // Scalar b = beta(gamma);
  // Scalar result = gamma * e1p * (1.0 - b);
  // result += gamma * b * (1.0 - e1p / ep);
  // if (result > gamma - 1.0) result = gamma - 1.0;
  // if (result < 1.0e-5) result = 1.0e-5;
  // if (n == 33 && result / gamma < 0.2)
  //   Logger::print_info("e1p is {}, ep is {}, mu is {}, result is {}",
  //                      e1p, ep, mu, result / gamma);
  float u = m_dist(m_generator);
  Scalar bb, result;
  int gn = find_n_gamma(gamma);
  // if (gamma < LOW_GAMMA_THR) {
  //   Scalar l, h;
  //   int b = binary_search(u, gn, m_dNde_thompson, l, h);
  //   bb = (l == h ? b : (u - l) / (h - l) + b);
  //   result = exp(m_dlep * bb + log(MIN_EP));
  // } else {
  Scalar l, h;
  int b = binary_search(u, gn, m_dNde, l, h);
  if (b < 2) {
    b = binary_search(u, gn, m_dNde_thompson, l, h);
    bb = (l == h ? b : (u - l) / (h - l) + b);
    result = exp(m_dlep * bb + log(MIN_EP));
  } else {
    bb = (l == h ? b : (u - l) / (h - l) + b);
    result = m_dep * bb;
  }
  // }

  // Scalar result = m_dep * bb;
  return result * gamma;
}

void
inverse_compton::generate_photon_energies(cu_array<Scalar>& e_ph,
                                          cu_array<Scalar>& gammas) {
  Kernels::gen_photon_energies<<<m_blocks, m_threads>>>(
      e_ph.data_d(), gammas.data_d(), gammas.size(),
      (curandState*)m_states);
  CudaCheckError();
}

void
inverse_compton::generate_random_gamma(cu_array<Scalar>& gammas) {
  Kernels::gen_rand_gammas<<<m_blocks, m_threads>>>(
      gammas.data_d(), gammas.size(), (curandState*)m_states);
  CudaCheckError();
}

template void inverse_compton::init<Spectra::power_law_hard>(
    const Spectra::power_law_hard& n_e, Scalar emin, Scalar emax,
    double n0);
template void inverse_compton::init<Spectra::power_law_soft>(
    const Spectra::power_law_soft& n_e, Scalar emin, Scalar emax,
    double n0);
template void inverse_compton::init<Spectra::black_body>(
    const Spectra::black_body& n_e, Scalar emin, Scalar emax,
    double n0);
template void inverse_compton::init<Spectra::mono_energetic>(
    const Spectra::mono_energetic& n_e, Scalar emin, Scalar emax,
    double n0);
template void inverse_compton::init<Spectra::broken_power_law>(
    const Spectra::broken_power_law& n_e, Scalar emin, Scalar emax,
    double n0);

}  // namespace Aperture
