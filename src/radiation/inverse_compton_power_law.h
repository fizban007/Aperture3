#ifndef _INVERSE_COMPTON_PL_H_
#define _INVERSE_COMPTON_PL_H_

// #include "data/photons.h"
// #include "data/particles.h"
// #include "data/fields.h"
#include "data/typedefs.h"
#include "cuda/cuda_control.h"

namespace Aperture {

template <typename RandFunc>
class InverseComptonPL
{
 public:
  HOST_DEVICE InverseComptonPL(Scalar alpha, Scalar e_s, Scalar e_min, Scalar mfp, RandFunc& rng);
  HOST_DEVICE ~InverseComptonPL();

  // Draw the lab frame photon energy
  HOST_DEVICE Scalar draw_photon_energy(Scalar gamma, Scalar p);
  // Draw the lab frame photon free path
  HOST_DEVICE Scalar draw_photon_freepath(Scalar Eph);

 private:
  Scalar m_alpha, m_es, m_emin, m_mfp;
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
}; // ----- end of class InverseComptonPL -----


template <typename RandFunc>
InverseComptonPL<RandFunc> make_inverse_compton_PL(Scalar alpha, Scalar e_s, Scalar e_min, Scalar mfp, RandFunc& rng);

}

#include "radiation/inverse_compton_power_law.cuh"

#endif  // _INVERSE_COMPTON_PL_H_
