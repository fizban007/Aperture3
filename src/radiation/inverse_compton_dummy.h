#ifndef _INVERSE_COMPTON_DUMMY_H_
#define _INVERSE_COMPTON_DUMMY_H_

#include "cuda/cuda_control.h"
#include "data/typedefs.h"
#include "sim_params.h"

namespace Aperture {

template <typename RandFunc>
class InverseComptonDummy {
 public:
  HOST_DEVICE InverseComptonDummy(const SimParamsBase& params,
                                  RandFunc& rng);
  HOST_DEVICE virtual ~InverseComptonDummy();

  // Determine whether a high energy photon should be emitted
  HOST_DEVICE bool emit_photon(Scalar gamma);
  // Draw the lab frame photon energy
  HOST_DEVICE Scalar draw_photon_energy(Scalar gamma, Scalar p);
  // Draw the lab frame photon free path
  HOST_DEVICE Scalar draw_photon_freepath(Scalar Eph);

 private:
  Scalar m_cutoff, m_Eph, m_lph;
  RandFunc& m_rng;
};  // ----- end of class InverseComptonDummy -----

template <typename RandFunc>
HOST_DEVICE InverseComptonDummy<RandFunc> make_inverse_compton_dummy(
    const SimParamsBase& params, RandFunc& rng);
}  // namespace Aperture

#include "radiation/inverse_compton_dummy.cuh"

#endif  // _INVERSE_COMPTON_DUMMY_H_
