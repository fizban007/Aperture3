#ifndef _INVERSE_COMPTON_PL_H_
#define _INVERSE_COMPTON_PL_H_

// #include "data/photons.h"
// #include "data/particles.h"
// #include "data/fields.h"
#include "data/typedefs.h"
#include "cuda/cuda_control.h"

// class curandState;

namespace Aperture {

class InverseComptonPL
{
 public:
  HOST_DEVICE InverseComptonPL(Scalar alpha, Scalar e_s, Scalar e_min, Scalar mfp);
  HOST_DEVICE ~InverseComptonPL();

  HOST_DEVICE Scalar draw_photon_energy(Scalar gamma, Scalar p, Scalar x);
  HOST_DEVICE Scalar draw_photon_freepath(Scalar Eph);

 private:
  Scalar m_alpha, m_es, m_emin, m_mfp;
  // void* d_rand_states;
  // int m_threadsPerBlock, m_blocksPerGrid;
}; // ----- end of class Inverse -----


}

#endif  // _INVERSE_COMPTON_PL_H_
