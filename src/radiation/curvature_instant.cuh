#ifndef _CURVATURE_INSTANT_CUH_
#define _CURVATURE_INSTANT_CUH_

#include "utils/logger.h"
#include "utils/util_functions.h"

namespace Aperture {

template <typename RandFunc>
HOST_DEVICE
CurvatureInstant<RandFunc>::CurvatureInstant(
    const SimParamsBase& params, RandFunc& rng)
    : m_thr(params.gamma_thr),
      m_E_sec(params.E_secondary),
      m_r_cutoff(params.r_cutoff),
      m_ph_mfp(params.photon_path),
      m_rng(rng) {
  // Logger::print_info("Gamma threshold is {}", m_thr);
  // Logger::print_info("E secondary is {}", m_E_sec);
  // Logger::print_info("Radius cutoff is {}", m_r_cutoff);
  // Logger::print_info("Photon free path is {}", m_ph_mfp);
}

template <typename RandFunc>
HOST_DEVICE CurvatureInstant<RandFunc>::~CurvatureInstant() {}

template <typename RandFunc>
HOST_DEVICE bool
CurvatureInstant<RandFunc>::emit_photon(Scalar gamma) {
  if (gamma > m_thr)
    return true;
  else
    return false;
}

template <typename RandFunc>
HOST_DEVICE Scalar
CurvatureInstant<RandFunc>::draw_photon_energy(Scalar gamma, Scalar p) {
  return 2.0f * m_E_sec;
}

template <typename RandFunc>
HOST_DEVICE Scalar
CurvatureInstant<RandFunc>::draw_photon_freepath(Scalar Eph) {
  float u = m_rng();
  return m_ph_mfp * std::sqrt(-2.0 * std::log(u));
}

}  // namespace Aperture

#endif  // _CURVATURE_INSTANT_CUH_