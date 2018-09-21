#ifndef _INVERSE_COMPTON_PL_H_
#define _INVERSE_COMPTON_PL_H_

// #include "data/photons.h"
// #include "data/particles.h"
// #include "data/fields.h"
#include "cuda/cuda_control.h"
#include "data/typedefs.h"
#include "sim_params.h"

namespace Aperture {

template <typename RandFunc>
class InverseComptonPL1D {
 public:
  HOST_DEVICE InverseComptonPL1D(const SimParamsBase& params,
                                 RandFunc& rng);
  HOST_DEVICE ~InverseComptonPL1D();

  // Determine whether a high energy photon should be emitted
  HOST_DEVICE bool emit_photon(Scalar gamma);
  // Draw the lab frame photon energy
  HOST_DEVICE Scalar draw_photon_energy(Scalar gamma, Scalar p);
  // Draw the lab frame photon free path
  HOST_DEVICE Scalar draw_photon_freepath(Scalar Eph);

 private:
  Scalar m_alpha, m_es, m_emin, m_mfp, m_icrate;
  RandFunc& m_rng;

  HOST_DEVICE Scalar compute_A1(Scalar er);
  HOST_DEVICE Scalar compute_A2(Scalar er, Scalar et);
  HOST_DEVICE Scalar f_inv1(float u, Scalar gamma);
  HOST_DEVICE Scalar f_inv2(float u, Scalar gamma);

  HOST_DEVICE Scalar draw_photon_e1p(Scalar gamma);
  HOST_DEVICE Scalar draw_photon_ep(Scalar e1p, Scalar gamma);
  HOST_DEVICE Scalar draw_photon_u1p(Scalar e1p, Scalar gamma);
  // void* d_rand_states;
  // int m_threadsPerBlock, m_blocksPerGrid;
};  // ----- end of class InverseComptonPL1D -----

template <typename RandFunc>
HOST_DEVICE InverseComptonPL1D<RandFunc> make_inverse_compton_PL1D(
    const SimParamsBase& params, RandFunc& rng);
}  // namespace Aperture

#include "radiation/inverse_compton_power_law.cuh"

#endif  // _INVERSE_COMPTON_PL_H_
