#ifndef _INVERSE_COMPTON_BB_H_
#define _INVERSE_COMPTON_BB_H_

// #include "data/photons.h"
// #include "data/particles.h"
// #include "data/fields.h"
#include "cuda/cuda_control.h"
#include "data/typedefs.h"

// class curandState;

namespace Aperture {

class InverseComptonBB {
 public:
  HOST_DEVICE InverseComptonBB(Scalar kT, Scalar n0 = 1.0f);
  HOST_DEVICE ~InverseComptonBB();

  HOST_DEVICE Scalar draw_photon_energy(Scalar u, Scalar gamma,
                                        Scalar p, Scalar x);
  HOST_DEVICE Scalar draw_photon_freepath(Scalar Eph);

 private:
  Scalar m_kT, m_n0;
};  // ----- end of class Inverse -----

}  // namespace Aperture

#endif  // _INVERSE_COMPTON_BB_H_
