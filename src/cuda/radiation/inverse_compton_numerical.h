#ifndef _INVERSE_COMPTON_NUMERICAL_H_
#define _INVERSE_COMPTON_NUMERICAL_H_

#include "data/typedefs.h"
#include "cuda/cuda_control.h"

namespace Aperture {

template <typename RandFunc>
class InverseComptonNumerical
{
 public:
  HOST_DEVICE InverseComptonNumerical(Scalar* n_e, Scalar* e, int size, RandFunc& rng);
  HOST_DEVICE ~InverseComptonNumerical();

  // Determine whether a high energy photon should be emitted
  HOST_DEVICE bool emit_photon(Scalar gamma);
  // Draw the lab frame photon energy
  HOST_DEVICE Scalar draw_photon_energy(Scalar gamma, Scalar p);
  // Draw the lab frame photon free path
  HOST_DEVICE Scalar draw_photon_freepath(Scalar Eph);

 private:
  RandFunc& m_rng;

};


}

#endif  // _INVERSE_COMPTON_NUMERICAL_H_
