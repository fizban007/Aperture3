#include "utils/util_functions.h"

namespace Aperture {

template <typename RandFunc>
HOST_DEVICE
InverseComptonDummy<RandFunc>::InverseComptonDummy(const SimParamsBase& params, RandFunc& rng)
    : m_cutoff(params.E_cutoff), m_Eph(params.E_ph),
      m_lph(params.photon_path), m_rng(rng) {}

template <typename RandFunc>
HOST_DEVICE
InverseComptonDummy<RandFunc>::~InverseComptonDummy() {}

template <typename RandFunc>
HOST_DEVICE
bool
InverseComptonDummy<RandFunc>::emit_photon(Scalar gamma) {
  return (gamma > m_cutoff);
}

template <typename RandFunc>
HOST_DEVICE
Scalar
InverseComptonDummy<RandFunc>::draw_photon_energy(Scalar gamma, Scalar p) {
  return m_Eph;
}

template <typename RandFunc>
HOST_DEVICE
Scalar
InverseComptonDummy<RandFunc>::draw_photon_freepath(Scalar Eph) {
  return m_lph;
}

template <typename RandFunc>
InverseComptonDummy<RandFunc> make_inverse_compton_Dummy(const SimParamsBase& params, RandFunc& rng) {
  InverseComptonDummy<RandFunc> result(params, rng);
  return result;
}


}