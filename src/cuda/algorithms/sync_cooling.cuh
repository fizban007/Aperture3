#ifndef _SYNC_COOLING_CUH_
#define _SYNC_COOLING_CUH_

#include "cuda/constant_mem.h"

namespace Aperture {

namespace Kernels {

__device__ __forceinline__ void
sync_cooling(Scalar& p1, Scalar& p2, Scalar& p3, Scalar& gamma,
             Scalar& B1, Scalar& B2, Scalar& B3, Scalar& E1, Scalar& E2,
             Scalar& E3, Scalar& q_over_m) {
  Scalar tmp1 = (E1 + (p2 * B3 - p3 * B2) / gamma) / q_over_m;
  Scalar tmp2 = (E2 + (p3 * B1 - p1 * B3) / gamma) / q_over_m;
  Scalar tmp3 = (E3 + (p1 * B2 - p2 * B1) / gamma) / q_over_m;
  Scalar tmp_sq = tmp1 * tmp1 + tmp2 * tmp2 + tmp3 * tmp3;
  Scalar bE = (p1 * E1 + p2 * E2 + p3 * E3) / (gamma * q_over_m);

  Scalar delta_p1 = dev_params.rad_cooling_coef *
                    (((tmp2 * B3 - tmp3 * B2) + bE * E1) / q_over_m -
                     gamma * p1 * (tmp_sq - bE * bE)) /
                    square(dev_params.B0);
  Scalar delta_p2 = dev_params.rad_cooling_coef *
                    (((tmp3 * B1 - tmp1 * B3) + bE * E2) / q_over_m -
                     gamma * p2 * (tmp_sq - bE * bE)) /
                    square(dev_params.B0);
  Scalar delta_p3 = dev_params.rad_cooling_coef *
                    (((tmp1 * B2 - tmp2 * B1) + bE * E3) / q_over_m -
                     gamma * p3 * (tmp_sq - bE * bE)) /
                    square(dev_params.B0);

  p1 += delta_p1;
  p2 += delta_p2;
  p3 += delta_p3;
  // p = sqrt(p1 * p1 + p2 * p2 + p3 * p3);
  gamma = sqrt(1.0f + p1 * p1 + p2 * p2 + p3 * p3);
}

__device__ __forceinline__ void
sync_kill_perp(Scalar& p1, Scalar& p2, Scalar& p3, Scalar& gamma,
               Scalar B1, Scalar B2, Scalar B3, Scalar& E1,
               Scalar& E2, Scalar& E3, Scalar& q_over_m) {
  B1 /= q_over_m;
  B2 /= q_over_m;
  B3 /= q_over_m;
  Scalar p = sqrt(p1 * p1 + p2 * p2 + p3 * p3);
  Scalar B_sqr = B1 * B1 + B2 * B2 + B3 * B3;
  Scalar pdotB = (p1 * B1 + p2 * B2 + p3 * B3);

  Scalar delta_p1 = -dev_params.rad_cooling_coef *
                    (p1 - B1 * pdotB / B_sqr);
  Scalar delta_p2 = -dev_params.rad_cooling_coef *
                    (p2 - B2 * pdotB / B_sqr);
  Scalar delta_p3 = -dev_params.rad_cooling_coef *
                    (p3 - B3 * pdotB / B_sqr);
  // Scalar dp = sqrt(delta_p1 * delta_p1 + delta_p2 * delta_p2 +
  //                  delta_p3 * delta_p3);
  Scalar f = std::sqrt(B_sqr) / dev_params.B0;
  // Scalar f = B_sqr / square(dev_params.B0);
  // if (sp == (int)ParticleType::ion) f *= 0.1f;
  p1 += delta_p1 * f;
  p2 += delta_p2 * f;
  p3 += delta_p3 * f;
  p = sqrt(p1 * p1 + p2 * p2 + p3 * p3);
  gamma = sqrt(1.0f + p * p);
}

}

}  // namespace Aperture

#endif  // _SYNC_COOLING_H_
