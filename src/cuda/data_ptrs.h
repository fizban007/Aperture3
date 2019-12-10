#ifndef _DATA_PTRS_H_
#define _DATA_PTRS_H_

#include "cuda/utils/pitchptr.h"
#include "data/particle_data.h"

namespace Aperture {

struct sim_data;

struct data_ptrs {
  pitchptr<Scalar> E1, E2, E3;
  pitchptr<Scalar> Ebg1, Ebg2, Ebg3;
  pitchptr<Scalar> B1, B2, B3;
  pitchptr<Scalar> Bbg1, Bbg2, Bbg3;
  pitchptr<Scalar> J1, J2, J3;
  pitchptr<Scalar>* Rho;
  pitchptr<Scalar>* gamma;
  pitchptr<Scalar>* ptc_num;

  pitchptr<Scalar> divE, divB, EdotB;
  pitchptr<Scalar> photon_produced;
  pitchptr<Scalar> pair_produced;
  pitchptr<Scalar> photon_num;
  pitchptr<float> ph_flux;

  particle_data particles;
  photon_data photons;
};

data_ptrs get_data_ptrs(sim_data& data);

}  // namespace Aperture

#endif  // _DATA_PTRS_H_
