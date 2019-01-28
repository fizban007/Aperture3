#ifndef _INVERSE_COMPTON_BB_H_
#define _INVERSE_COMPTON_BB_H_

#include "cuda/cuda_control.h"
#include "core/typedefs.h"
#include "sim_params.h"

// class curandState;

namespace Aperture {

template <typename RandFunc>
class InverseComptonBB {
 public:
  HOST_DEVICE InverseComptonBB(const SimParamsBase& params,
                               RandFunc& rng);
  HOST_DEVICE ~InverseComptonBB();

  // Determine whether a high energy photon should be emitted
  HOST_DEVICE bool emit_photon(Scalar gamma);
  // Draw the lab frame photon energy
  HOST_DEVICE Scalar draw_photon_energy(Scalar gamma, Scalar p);
  // Draw the lab frame photon free path
  HOST_DEVICE Scalar draw_photon_freepath(Scalar Eph);

 private:
  Scalar m_kT, m_n0;
  RandFunc& m_rng;

  HOST_DEVICE Scalar draw_photon_e1p(Scalar gamma);
  HOST_DEVICE Scalar draw_photon_ep(Scalar e1p, Scalar gamma);
  HOST_DEVICE Scalar draw_photon_u1p(Scalar e1p, Scalar gamma);
};  // ----- end of class Inverse -----


}  // namespace Aperture

#include "radiation/inverse_compton_black_body.cuh"

#endif  // _INVERSE_COMPTON_BB_H_
