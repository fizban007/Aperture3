#include "utils/util_functions.h"
// #include "cuda/cudaUtility.h"
// #include "cuda/constant_mem.h"

namespace Aperture {

template <typename RandFunc>
HOST_DEVICE Scalar
InverseComptonPL1D<RandFunc>::compute_A1(Scalar er) {
  Scalar A1 = 1.0 / (er * (0.5 + 1.0 / m_alpha -
                           (1.0 / (m_alpha * (m_alpha + 1.0))) *
                               std::pow(er / m_es, m_alpha)));
  return A1;
}

template <typename RandFunc>
HD_INLINE Scalar
InverseComptonPL1D<RandFunc>::compute_A2(Scalar er, Scalar et) {
  Scalar A2 = 1.0 / (et * (et * 0.5 / er + std::log(er / et) +
                           1.0 / (1.0 + m_alpha)));
  return A2;
}

template <typename RandFunc>
HOST_DEVICE Scalar
InverseComptonPL1D<RandFunc>::f_inv1(float u, Scalar gamma) {
  // Scalar alpha = dev_params.spectral_alpha;
  // Scalar e_s = dev_params.e_s;
  Scalar er = 2.0 * gamma * m_emin;
  Scalar A1 = compute_A1(er);
  if (u < A1 * er * 0.5)
    return std::sqrt(2.0 * u * er / A1);
  else if (u < 1.0 - A1 * er * std::pow(m_es / er, -m_alpha) /
                         (1.0 + m_alpha))
    return er *
           std::pow(m_alpha * (1.0 / m_alpha + 0.5 - u / (A1 * er)),
                    -1.0 / m_alpha);
  else
    return er * std::pow((1.0 - u) * (1.0 + m_alpha) / (A1 * m_es),
                         -1.0 / (m_alpha + 1.0));
}

template <typename RandFunc>
HOST_DEVICE Scalar
InverseComptonPL1D<RandFunc>::f_inv2(float u, Scalar gamma) {
  // Scalar alpha = dev_params.spectral_alpha;
  // Scalar e_s = dev_params.e_s;
  Scalar er = 2.0 * gamma * m_emin;
  Scalar et = er / (2.0 * er + 1.0);
  Scalar A2 = compute_A2(er, et);
  if (u < A2 * et * et * 0.5 / er)
    return std::sqrt(2.0 * u * er / A2);
  else if (u < 1.0 - A2 * et / (1.0 + m_alpha))
    return et * std::exp(u / (A2 * et) - et * 0.5 / er);
  else
    return er * std::pow((1.0 - u) * (1.0 + m_alpha) / (A2 * et),
                         -1.0 / (m_alpha + 1.0));
}

// Draw the rest frame photon energy
template <typename RandFunc>
HOST_DEVICE Scalar
InverseComptonPL1D<RandFunc>::draw_photon_e1p(Scalar gamma) {
  float u = m_rng();
  Scalar e1p;
  if (gamma < m_es * 0.5 / m_emin) {
    e1p = f_inv1(u, gamma);
  } else {
    e1p = f_inv2(u, gamma);
  }
  return e1p;
}

// Given rest frame photon energy, draw its original energy
template <typename RandFunc>
HOST_DEVICE Scalar
InverseComptonPL1D<RandFunc>::draw_photon_ep(Scalar e1p, Scalar gamma) {
  float u = m_rng();
  double ep;
  double gemin2 = 2.0 * gamma * m_emin;
  // Scalar alpha = dev_params.spectral_alpha;
  if (e1p < 0.5 && e1p / (1.0 - 2.0 * e1p) <= gemin2) {
    double e_lim = e1p / (1.0 - 2.0 * e1p);
    double a1 = (gemin2 * gemin2 * (m_alpha + 2.0)) /
                (gamma * (e_lim * e_lim - e1p * e1p));
    ep =
        std::sqrt(u * (m_alpha + 2.0) * gemin2 * gemin2 / (a1 * gamma) +
                  e1p * e1p);
  } else if (e1p > gemin2) {
    double a2 = (m_alpha * (m_alpha + 2.0) * 0.5 / gamma) *
                std::pow(e1p / gemin2, m_alpha);
    if (e1p < 0.5) a2 /= (1.0 - std::pow(1.0 - 2.0 * e1p, m_alpha));
    ep = gemin2 * std::pow(std::pow(gemin2 / e1p, m_alpha) -
                               u * m_alpha * (m_alpha + 2.0) /
                                   (2.0 * gamma * a2),
                           -1.0 / m_alpha);
  } else {
    double G = 0.0;
    if (e1p < 0.5)
      G = std::pow((1.0 - 2.0 * e1p) * gemin2 / e1p, m_alpha);
    double U_0 = (gemin2 * gemin2 - e1p * e1p) * gamma /
                 (gemin2 * gemin2 * (m_alpha + 2.0));
    double a3 = 1.0 / (U_0 + (1.0 - G) * 2.0 * gamma /
                                 (m_alpha * (m_alpha + 2.0)));
    if (u < U_0 * a3)
      ep = std::sqrt(u * (m_alpha + 2.0) * gemin2 * gemin2 /
                         (a3 * gamma) +
                     e1p * e1p);
    else
      ep = gemin2 *
           std::pow(1.0 - (u - a3 * U_0) * m_alpha * (m_alpha + 2.0) /
                              (2.0 * a3 * gamma),
                    -1.0 / m_alpha);
  }
  return ep;
}

// Given rest frame photon energy, draw the rest frame photon angle
template <typename RandFunc>
HOST_DEVICE Scalar
InverseComptonPL1D<RandFunc>::draw_photon_u1p(Scalar e1p,
                                              Scalar gamma) {
  Scalar u1p;
  Scalar ep = draw_photon_ep(e1p, gamma);
  u1p = 1.0 - 1.0 / e1p + 1.0 / ep;
  return u1p;
}

template <typename RandFunc>
HOST_DEVICE
InverseComptonPL1D<RandFunc>::InverseComptonPL1D(
    const SimParamsBase& params, RandFunc& rng)
    : m_alpha(params.spectral_alpha),
      m_es(params.e_s),
      m_emin(params.e_min),
      m_mfp(params.photon_path),
      m_rng(rng) {}

template <typename RandFunc>
HOST_DEVICE InverseComptonPL1D<RandFunc>::~InverseComptonPL1D() {}

template <typename RandFunc>
HOST_DEVICE bool
InverseComptonPL1D<RandFunc>::emit_photon(Scalar gamma) {
  float u = m_rng();
  // TODO: Finish photon emission rate
  float rate = (gamma * m_emin < 0.1 ? 0.1 : 0.01);
  return (u < rate);
}

template <typename RandFunc>
HOST_DEVICE Scalar
InverseComptonPL1D<RandFunc>::draw_photon_energy(Scalar gamma,
                                                 Scalar p) {
  Scalar e1p = draw_photon_e1p(gamma);
  Scalar u1p = draw_photon_u1p(e1p, gamma);

  return sgn(p) * (gamma + std::abs(p) * (-u1p)) * e1p;
}

template <typename RandFunc>
HOST_DEVICE Scalar
InverseComptonPL1D<RandFunc>::draw_photon_freepath(Scalar Eph) {
  Scalar rate;
  if (Eph * m_emin < 2.0) {
    rate = std::pow(Eph * m_emin / 2.0, m_alpha);
  } else {
    // rate = std::pow(Eph * e_min / 2.0, -1.0);
    rate = 2.0 / (Eph * m_emin);
  }

  return -m_mfp * std::log(1.0f - m_rng()) / rate;
}

template <typename RandFunc>
HOST_DEVICE InverseComptonPL1D<RandFunc>
make_inverse_compton_PL1D(const SimParamsBase& params, RandFunc& rng) {
  InverseComptonPL1D<RandFunc> result(params, rng);
  return result;
}

}  // namespace Aperture