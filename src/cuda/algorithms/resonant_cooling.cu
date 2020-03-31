#ifndef _RESONANT_COOLING_CUH_
#define _RESONANT_COOLING_CUH_

#include "cuda/constant_mem.h"
#include "utils/util_functions.h"

namespace Aperture {

namespace Kernels {

__device__ __forceinline__ Scalar
resonant_cooling(Scalar& p1, Scalar& p2, Scalar& p3, Scalar& gamma,
                 uint32_t& flag, Scalar r, Scalar pdotB, Scalar B,
                 Scalar B1, Scalar B2, Scalar B3, Scalar E1, Scalar E2,
                 Scalar E3, Scalar q_over_m, Scalar dt) {
  // Scalar B = sqrt(B1 * B1 + B2 * B2 + B3 * B3) / std::abs(q_over_m);
  Scalar gamma_thr_B = dev_params.gamma_thr * B / dev_params.BQ;

  if (gamma_thr_B > 3.0f && gamma > gamma_thr_B) {
    flag = flag |= bit_or(ParticleFlag::emit_photon);
  } else if (dev_params.rad_cooling_on) {
    // Process resonant drag
    Scalar p_mag_signed = sgn(pdotB) * sgn(B1) * std::abs(pdotB) / B;
    // printf("p_mag_signed is %f\n", p_mag_signed);
    Scalar g = sqrt(1.0f + p_mag_signed * p_mag_signed);
    Scalar mu = std::abs(B1 / B);
    Scalar y = (B / dev_params.BQ) /
               (dev_params.star_kT * (g - p_mag_signed * mu));
    // printf("g is %f, y is %f\n", g, y);
    if (y < 20.0f) {
      // printf("y is %f\n", y);
      Scalar coef = dev_params.res_drag_coef * y * y * y /
                    (r * r * (std::exp(y) - 1.0f));
      // printf("coef is %f\n", coef);
      // printf("drag coef is %f\n", dev_params.res_drag_coef);
      Scalar D = coef * (g * mu - p_mag_signed);
      if (B1 < 0.0f) D *= -1.0f;
      // printf("D is %f\n", D);
      p1 += dt * B1 * D / B;
      p2 += dt * B2 * D / B;
      p3 += dt * B3 * D / B;
      // printf("drag on p1 is %f\n", dt * B1 * D / B);
      gamma = sqrt(1.0f + p1 * p1 + p2 * p2 + p3 * p3);

      Scalar Ndot = std::abs(coef * (1.0f - p_mag_signed * mu / g));
      return Ndot;
      // Scalar angle =
      //     acos(sgn(pdotB) * (B1 * cos(theta) - B2 * sin(theta)) / B);
      // // Scalar theta_p =
      // //     2.0f * CONST_PI * curand_uniform(&localState);
      // Scalar theta_p = CONST_PI * curand_uniform(&localState);
      // Scalar phi_p = 2.0f * CONST_PI * curand_uniform(&localState);
      // Scalar u = std::cos(theta_p);
      // Scalar beta = sqrt(1.0f - 1.0f / square(g));
      // // angle = angle + sgn(theta_p - CONST_PI) *
      // //                     std::acos((u + beta) / (1.0f + beta *
      // //                     u));
      // // angle = angle + (2.0f*phi_p - 1.0f) * std::acos((u + beta)
      // // / (1.0f + beta * u));
      // Scalar cos_angle =
      //     std::cos(angle) * std::cos(theta_p) +
      //     std::sin(angle) * std::sin(theta_p) * std::cos(phi_p);
      // // angle =
      // angle = std::acos(cos_angle);
      // Scalar Eph =
      //     (g - std::abs(p_mag_signed) * u) *
      //     (1.0f - 1.0f / sqrt(1.0f + 2.0f * B / dev_params.BQ));
      // if (p1 > 0.0f && (Eph < 2.0f || B < 0.1 * dev_params.BQ)) {
      //   Eph = std::log(Eph) / std::log(10.0f);
      //   if (Eph > 2.0f) Eph = 2.0f;
      //   if (Eph < -6.0f) Eph = -6.0f;
      //   int n0 = ((Eph + 6.0f) / 8.02f *
      //             (ph_flux.xsize / sizeof(float) - 1));
      //   if (n0 < 0) n0 = 0;
      //   if (n0 >= ph_flux.xsize / sizeof(float))
      //     n0 = ph_flux.xsize / sizeof(float) - 1;
      //   int n1 = (std::abs(angle) / (CONST_PI + 1.0e-5)) *
      //            (ph_flux.ysize - 1);
      //   if (n1 < 0) n1 = 0;
      //   if (n1 >= ph_flux.ysize) n1 = ph_flux.ysize - 1;
      //   auto w = ptc.weight[idx];
      //   // printf("n0 is %d, n1 is %d, Ndot is %f\n", n0, n1, Ndot);
      //   atomicAdd(ptrAddr(ph_flux, n0, n1), Ndot * dt * w);
      // }
    }
  }
  return 0.0f;
}

}  // namespace Kernels

}  // namespace Aperture

#endif  // _RESONANT_COOLING_CUH_
