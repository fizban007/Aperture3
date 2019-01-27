#ifndef _CURVATURE_INSTANT_H_
#define _CURVATURE_INSTANT_H_

#include "cuda/cuda_control.h"
#include "core/typedefs.h"
#include "sim_params.h"

namespace Aperture {

template <typename RandFunc>
class CurvatureInstant {
 public:
  HOST_DEVICE CurvatureInstant(const SimParamsBase& params,
                               RandFunc& rng);
  HOST_DEVICE ~CurvatureInstant();

  // Determine whether a high energy photon should be emitted
  HOST_DEVICE bool emit_photon(Scalar gamma);
  // Draw the lab frame photon energy
  HOST_DEVICE Scalar draw_photon_energy(Scalar gamma, Scalar p);
  // Draw the lab frame photon free path
  HOST_DEVICE Scalar draw_photon_freepath(Scalar Eph);
  
 private:
  Scalar m_thr, m_E_sec, m_r_cutoff, m_ph_mfp;
  RandFunc& m_rng;
};

}

#include "radiation/curvature_instant.cuh"

#endif  // _CURVATURE_INSTANT_H_
