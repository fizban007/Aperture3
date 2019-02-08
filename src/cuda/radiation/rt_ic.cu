#include "cuda/constant_mem.h"
#include "cuda/cudaUtility.h"
#include "cuda/ptr_util.h"
#include "cuda/radiation/rt_ic.h"
#include "radiation/spectra.h"
#include "sim_params.h"
#include "utils/logger.h"
#include "utils/util_functions.h"
#include <curand_kernel.h>

namespace Aperture {

#define MIN_GAMMA 1.01f
#define MAX_GAMMA 1.0e12
#define MIN_EP 1.0e-15
#define MAX_EP 1.0e15

namespace Kernels {

__constant__ Scalar dev_ic_dep;
__constant__ Scalar dev_ic_dg;

__device__ int
find_n_gamma(Scalar gamma) {
  if (gamma < MIN_GAMMA) return 0;
  if (gamma > MAX_GAMMA) return dev_params.n_gamma - 1;
  return (std::log(gamma) - std::log(MIN_GAMMA)) / dev_ic_dg;
}

__device__ Scalar
gen_e1p(Scalar gamma, curandState& state, cudaPitchedPtr dnde1p) {
  int n = find_n_gamma(gamma);
  float u = curand_uniform(&state);
  int a = 0, b = dev_params.n_ep;
  while (a < b) {
    int tmp = (a + b) / 2;
    // size_t offset = tmp * sizeof(Scalar) + n * dnde1p.pitch;
    Scalar v = *ptrAddr(dnde1p, tmp, n);
    if (v < u) {
      a = tmp + 1;
    } else if (v > u) {
      b = tmp - 1;
    } else {
      b = tmp;
      break;
    }
  }
  return std::exp(log(MIN_EP) + dev_ic_dep * b);
}

}  // namespace Kernels

inverse_compton::inverse_compton(const SimParams& params)
    : m_npep(Extent(params.n_ep, params.n_gamma)),
      m_dnde1p(Extent(params.n_ep, params.n_gamma)),
      m_rate(params.n_gamma),
      m_gammas(params.n_gamma),
      m_ep(params.n_ep),
      m_generator(),
      m_dist(0.0, 1.0) {
  m_dep = (log(MAX_EP) - log(MIN_EP)) / ((Scalar)m_ep.size() - 1.0);
  for (uint32_t n = 0; n < m_ep.size(); n++) {
    m_ep[n] = exp(log(MIN_EP) + m_dep * (Scalar)n);
  }
  m_dg =
      (log(MAX_GAMMA) - log(MIN_GAMMA)) / ((Scalar)m_rate.size() - 1.0);
  for (uint32_t n = 0; n < m_rate.size(); n++) {
    m_gammas[n] = exp(log(MIN_GAMMA) + m_dg * (Scalar)n);
  }

  CudaSafeCall(
      cudaMemcpyToSymbol(Kernels::dev_ic_dep, &m_dep, sizeof(Scalar)));
  CudaSafeCall(
      cudaMemcpyToSymbol(Kernels::dev_ic_dg, &m_dg, sizeof(Scalar)));
}

inverse_compton::~inverse_compton() {}

template <typename F>
void
inverse_compton::init(const F& n_e, Scalar emin, Scalar emax) {
  const int N_mu = 100;
  const int N_e = 500;

  // Compute the gammas and rates for IC scattering
  Logger::print_info("Pre-calculating the scattering rate");
  Scalar dmu = 2.0 / (N_mu - 1.0);
  Scalar de = (log(emax) - log(emin)) / (N_e - 1.0);
  for (uint32_t n = 0; n < m_rate.size(); n++) {
    Scalar gamma = m_gammas[n];
    // Logger::print_info("gamma is {}", gamma);
    Scalar result = 0.0;
    for (int i_mu = 0; i_mu < N_mu; i_mu++) {
      Scalar mu = i_mu * dmu - 1.0;
      for (int i_e = 0; i_e < N_e; i_e++) {
        Scalar e = exp(log(emin) + i_e * de);
        // integrand(i_e, i_mu)
        Scalar x = x_ic(gamma, e, mu);
        Scalar sigma = sigma_ic(x);
        // if (i_mu == 50)
        // Logger::print_info("e is {}, x is {}, sigma is {}", e, x,
        //                    sigma);
        result += 0.5f * n_e(e) * sigma * (1.0f - beta(gamma) * mu) * e;
      }
    }
    // Logger::print_info("gamma is {}, result is {}", gamma, result);
    m_rate[n] = result * dmu * de;
  }

  // Compute the photon spectrum in electron rest frame for various
  // gammas
  Logger::print_info(
      "Pre-calculating the rest frame soft photon spectrum");
  for (uint32_t n = 0; n < m_gammas.size(); n++) {
    Scalar gamma = m_gammas[n];
    for (uint32_t i = 0; i < m_ep.size(); i++) {
      Scalar ep = m_ep[i];
      Scalar result = 0.0;
      for (int i_e = 0; i_e < N_e; i_e++) {
        Scalar e = exp(log(emin) + i_e * de);
        if (e > 0.5 * ep / gamma && e < 2.0 * ep * gamma)
          result += n_e(e) * ep / (2.0 * gamma * e);
      }
      m_npep(i, n) = result * de;
    }
  }

  // Compute the scattered photon spectrum in the electron rest
  // frame. We only store the cumulative distribution for Monte Carlo
  // purpose
  Logger::print_info(
      "Pre-calculating the rest frame scattered photon spectrum");
  for (uint32_t n = 0; n < m_gammas.size(); n++) {
    for (uint32_t i = 0; i < m_ep.size(); i++) {
      Scalar e1p = m_ep[i];
      if (e1p < 0.03) {
        m_dnde1p(i, n) =
            m_npep(i, n) * 2.0 / (1.0 - 2.0 * e1p) * m_ep[i];
      } else {
        Scalar result = 0.0;
        for (uint32_t i_e = 0; i_e < m_ep.size(); i_e++) {
          Scalar ep = m_ep[i_e];
          if (ep > e1p && 1.0 / (1.0 / ep + 2.0) < e1p)
            result += m_npep(i_e, n) * sigma_rest(ep, e1p) / ep;
        }
        m_dnde1p(i, n) = result * m_dep * m_ep[i];
        // m_dnde1p(i, n) = result * m_dep;
      }
    }
    for (uint32_t i = 1; i < m_ep.size(); i++) {
      m_dnde1p(i, n) += m_dnde1p(i - 1, n);
    }
    for (uint32_t i = 0; i < m_ep.size(); i++) {
      m_dnde1p(i, n) /= m_dnde1p(m_ep.size() - 1, n);
    }
  }

  // Process the npep distribution to turn it into a cumulative
  // distribution
  for (uint32_t n = 0; n < m_gammas.size(); n++) {
    for (uint32_t i = 0; i < m_ep.size(); i++) {
      m_npep(i, n) *= m_ep[i];
    }
    for (uint32_t i = 1; i < m_ep.size(); i++) {
      m_npep(i, n) += m_npep(i - 1, n);
    }
    for (uint32_t i = 0; i < m_ep.size(); i++) {
      m_npep(i, n) /= m_npep(m_ep.size() - 1, n);
    }
  }

  m_gammas.sync_to_device();
  m_rate.sync_to_device();
  m_ep.sync_to_device();
  m_npep.sync_to_device();
  m_dnde1p.sync_to_device();
}

HOST_DEVICE double
inverse_compton::sigma_ic(Scalar x) const {
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
inverse_compton::x_ic(Scalar gamma, Scalar e, Scalar mu) const {
  return gamma * e * (1.0 - mu * beta(gamma));
}

HOST_DEVICE double
inverse_compton::beta(Scalar gamma) const {
  return 1.0 / sqrt(1.0 - 1.0 / square(gamma));
}

HOST_DEVICE double
inverse_compton::sigma_rest(Scalar ep, Scalar e1p) const {
  return (ep / e1p + e1p / ep -
          (1.0 - square(1.0 - 1.0 / e1p + 1.0 / ep)));
}

int
inverse_compton::find_n_gamma(Scalar gamma) const {
  if (gamma < MIN_GAMMA) return 0;
  if (gamma > MAX_GAMMA) return m_gammas.size() - 1;
  return (std::log(gamma) - std::log(MIN_GAMMA)) / m_dg;
}

int
inverse_compton::find_n_ep(Scalar ep) const {
  if (ep < MIN_EP) return 0;
  if (ep > MAX_EP) return m_ep.size() - 1;
  return (std::log(ep) - std::log(MIN_EP)) / m_dep;
}

int
inverse_compton::binary_search(
    float u, int n, const cu_multi_array<Scalar>& array) const {
  int a = 0, b = m_ep.size() - 1;
  while (a < b) {
    int tmp = (a + b) / 2;
    Scalar v = array(tmp, n);
    if (v < u) {
      a = tmp + 1;
    } else if (v > u) {
      b = tmp - 1;
    } else {
      b = tmp;
      break;
    }
  }
  return b;
}

Scalar
inverse_compton::gen_e1p(int gn) {
  // int n = find_n_gamma(gamma);
  float u = m_dist(m_generator);
  int b = binary_search(u, gn, m_dnde1p);
  Scalar l = m_dnde1p(b, gn);
  Scalar h = m_dnde1p(b+1, gn);
  // Logger::print_info("u is {}, b is {}, b+1 is {}", u, ,
  //                    m_dnde1p(b + 1, gn));
  Scalar bb = (u - l) / (h - l) + b;
  return std::exp(log(MIN_EP) + m_dep * bb);
}

Scalar
inverse_compton::gen_ep(int gn, Scalar e1p) {
  if (e1p < 0.01f) return e1p;
  Scalar u_low = 0.0, u_hi = 1.0;
  if (e1p > 0.5) {
    Scalar e_low = e1p;
    u_low = m_npep(find_n_ep(e_low), gn);
  } else {
    Scalar e_low = e1p, e_hi = e1p / (1.0 - 2.0 * e1p);
    u_low = m_npep(find_n_ep(e_low), gn);
    u_hi = m_npep(find_n_ep(e_hi), gn);
  }
  float u = m_dist(m_generator);
  u = u * (u_hi - u_low) + u_low;
  int b = binary_search(u, gn, m_npep);
  Scalar l = m_npep(b, gn);
  Scalar h = m_npep(b+1, gn);
  Scalar bb = (u - l) / (h - l) + b;
  return std::exp(log(MIN_EP) + m_dep * bb);
}

Scalar
inverse_compton::gen_photon_e(Scalar gamma) {
  int n = find_n_gamma(gamma);
  Scalar e1p = gen_e1p(n);
  Scalar ep = gen_ep(n, e1p);
  if (ep < e1p) ep = e1p;
  Scalar mu = 1.0 - 1.0 / e1p + 1.0 / ep;
  Scalar result = gamma * e1p * (1.0 + beta(gamma) * (-mu));
  if (result > gamma - 1.0) result = gamma - 1.0;
  if (result < 0.001 * gamma) result = 0.001 * gamma;
  return result;
}

template void inverse_compton::init<Spectra::power_law_hard>(
    const Spectra::power_law_hard& n_e, Scalar emin, Scalar emax);
template void inverse_compton::init<Spectra::power_law_soft>(
    const Spectra::power_law_soft& n_e, Scalar emin, Scalar emax);
template void inverse_compton::init<Spectra::black_body>(
    const Spectra::black_body& n_e, Scalar emin, Scalar emax);

}  // namespace Aperture
