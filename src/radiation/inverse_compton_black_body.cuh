#include "utils/logger.h"
#include "utils/util_functions.h"

namespace Aperture {

template <typename RandFunc>
InverseComptonBB<RandFunc>::InverseComptonBB(
    const SimParamsBase& params, RandFunc& rng)
    : m_rng(rng) {}

template <typename RandFunc>
InverseComptonBB<RandFunc>::~InverseComptonBB() {}

template <typename RandFunc>
HOST_DEVICE bool
InverseComptonBB<RandFunc>::emit_photon(Scalar gamma) {
  float u = m_rng();
  // TODO: Finish photon emission rate
  float rate = 0.1;
  // float rate = (e_p < 0.1f ? m_icrate : m_icrate * 0.1f / e_p);
  return (u < rate);
}

template <typename RandFunc>
HOST_DEVICE Scalar
InverseComptonBB<RandFunc>::draw_photon_energy(Scalar gamma, Scalar p) {
  Scalar e1p = draw_photon_e1p(gamma);
  Scalar u1p = draw_photon_u1p(e1p, gamma);

  return sgn(p) * (gamma + std::abs(p) * (-u1p)) * e1p;
}

template <typename RandFunc>
HOST_DEVICE Scalar
InverseComptonBB<RandFunc>::draw_photon_freepath(Scalar Eph) {
  Scalar rate = 0.1;

  return 1.0 / rate;
}

// Draw the rest frame photon energy
template <typename RandFunc>
HOST_DEVICE Scalar
InverseComptonBB<RandFunc>::draw_photon_e1p(Scalar gamma) {
  float u = m_rng();
  Scalar e1p;
  return e1p;
}

// Given rest frame photon energy, draw its original energy
template <typename RandFunc>
HOST_DEVICE Scalar
InverseComptonBB<RandFunc>::draw_photon_ep(Scalar e1p, Scalar gamma) {
  float u = m_rng();
  Scalar ep;
  // Scalar alpha = dev_params.spectral_alpha;
  return ep;
}

// Given rest frame photon energy, draw the rest frame photon angle
template <typename RandFunc>
HOST_DEVICE Scalar
InverseComptonBB<RandFunc>::draw_photon_u1p(Scalar e1p, Scalar gamma) {
  Scalar u1p;
  Scalar ep = draw_photon_ep(e1p, gamma);
  u1p = 1.0f - 1.0f / e1p + 1.0f / ep;
  return u1p;
}

}  // namespace Aperture