#include "cuda/radiation/rt_ic.h"
#include "sim_params.h"
#include "utils/logger.h"
#include "utils/util_functions.h"

namespace Aperture {

inverse_compton::inverse_compton(const SimParams& params)
    : m_npep(Extent(params.n_gamma, params.n_ep)),
      m_dnde1p(Extent(params.n_gamma, params.n_ep)),
      m_rate(params.n_gamma),
      m_gammas(params.n_gamma),
      m_ep(params.n_ep) {
  for (uint32_t n = 0; n < m_ep.size(); n++) {
    m_dep = (log(1.0e15) - log(1.0e-15)) / ((Scalar)m_ep.size() - 1.0);
    m_ep[n] = exp(log(1.0e-15) + m_dep * (Scalar)n);
  }
}

inverse_compton::~inverse_compton() {}

template <typename F>
void
inverse_compton::init(const F& n_e, Scalar emin, Scalar emax) {
  const int N_mu = 100;
  const int N_e = 500;

  // Compute the gammas and rates for IC scattering
  Scalar dmu = 2.0 / (N_mu - 1.0);
  Scalar de = (log(emax) - log(emin)) / (N_e - 1.0);
  for (uint32_t n = 0; n < m_rate.size(); n++) {
    m_gammas[n] = exp(log(1.1) + (log(1.0e12) - log(1.1)) * (Scalar)n /
                                     ((Scalar)m_rate.size() - 1.0));
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
  for (uint32_t n = 0; n < m_gammas.size(); n++) {
    Scalar gamma = m_gammas[n];
    for (uint32_t i = 0; i < m_ep.size(); i++) {
      Scalar ep = m_ep[i];
      Scalar result = 0.0;
      for (int i_e = 0; i_e < N_e; i_e++) {
        Scalar e = exp(log(emin) + i_e * de);
        if (e > 0.5*ep/gamma && e < 2.0*ep*gamma)
          result += n_e(e) * ep / (2.0 * gamma * e);
      }
      m_npep(n, i) = result * de;
    }
  }
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

}  // namespace Aperture
