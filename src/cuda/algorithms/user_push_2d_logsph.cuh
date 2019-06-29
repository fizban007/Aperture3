#ifndef _USER_PUSH_2D_LOGSPH_CUH_
#define _USER_PUSH_2D_LOGSPH_CUH_

#include "cuda/constant_mem.h"
#include "cuda/data_ptrs.h"
#include "gravity.cuh"
#include "sync_cooling.cuh"
#include "vay_push.cuh"

namespace Aperture {

namespace Kernels {

__device__ __forceinline__ void
user_push_2d_logsph(data_ptrs& data, size_t idx, Scalar dt) {
  auto& ptc = data.particles;

  auto c = ptc.cell[idx];
  // Skip empty particles
  if (c == MAX_CELL) return;
  int c1 = dev_mesh.get_c1(c);
  int c2 = dev_mesh.get_c2(c);
  if (!dev_mesh.is_in_bulk(c1, c2)) {
    ptc.cell[idx] = MAX_CELL;
    return;
  }

  // Load particle quantities
  Interpolator2D<spline_t> interp;
  auto flag = ptc.flag[idx];
  int sp = get_ptc_type(flag);
  auto old_x1 = ptc.x1[idx], old_x2 = ptc.x2[idx];
  auto p1 = ptc.p1[idx], p2 = ptc.p2[idx], p3 = ptc.p3[idx],
       gamma = ptc.E[idx];
  Scalar r = std::exp(dev_mesh.pos(0, c1, old_x1));
  Scalar alpha = alpha_gr(r);
  Scalar q_over_m = dt * 0.5f * dev_charges[sp] / dev_masses[sp];
  if (p1 != p1 || p2 != p2 || p3 != p3) {
    printf(
        "NaN detected in push! p1 is %f, p2 is %f, p3 is %f, gamma "
        "is %f\n",
        p1, p2, p3, gamma);
    asm("trap;");
    // p1 = p2 = p3 = 0.0f;
  }

  // step 0: Grab E & M fields at the particle position
  gamma = std::sqrt(1.0f + p1 * p1 + p2 * p2 + p3 * p3);
  if (!check_bit(flag, ParticleFlag::ignore_EM)) {
    Scalar E1 =
        alpha *
        (interp(data.E1, old_x1, old_x2, c1, c2, Stagger(0b110))) *
        // interp(dev_bg_fields.E1, old_x1, old_x2, c1, c2,
        //        Stagger(0b110))) *
        q_over_m;
    Scalar E2 =
        alpha *
        (interp(data.E2, old_x1, old_x2, c1, c2, Stagger(0b101))) *
        // interp(dev_bg_fields.E2, old_x1, old_x2, c1, c2,
        //        Stagger(0b101))) *
        q_over_m;
    Scalar E3 =
        alpha *
        (interp(data.E3, old_x1, old_x2, c1, c2, Stagger(0b011))) *
        // interp(dev_bg_fields.E3, old_x1, old_x2, c1, c2,
        //        Stagger(0b011))) *
        q_over_m;
    Scalar B1 =
        alpha *
        (interp(data.B1, old_x1, old_x2, c1, c2, Stagger(0b001)) +
         interp(data.Bbg1, old_x1, old_x2, c1, c2, Stagger(0b001))) *
        q_over_m;
    Scalar B2 =
        alpha *
        (interp(data.B2, old_x1, old_x2, c1, c2, Stagger(0b010)) +
         interp(data.Bbg2, old_x1, old_x2, c1, c2, Stagger(0b010))) *
        q_over_m;
    Scalar B3 =
        alpha *
        (interp(data.B3, old_x1, old_x2, c1, c2, Stagger(0b100))) *
        // interp(dev_bg_fields.B3, old_x1, old_x2, c1, c2,
        //        Stagger(0b100))) *
        q_over_m;

    // printf("B1 = %f, B2 = %f, B3 = %f\n", B1, B2, B3);
    // printf("E1 = %f, E2 = %f, E3 = %f\n", E1, E2, E3);
    // printf("B cell is %f\n", *ptrAddr(fields.B1, c1*sizeof(Scalar)
    // + c2*fields.B1.pitch)); printf("q over m is %f\n", q_over_m);
    // printf("gamma before is %f\n", gamma);
    // printf("p is (%f, %f, %f), gamma is %f\n", p1, p2, p3, gamma);
    vay_push(p1, p2, p3, gamma, E1, E2, E3, B1, B2, B3, q_over_m, dt);

    gravity(p1, p2, p3, gamma, r, dt);
    // printf("p after is (%f, %f, %f), gamma is %f, inv_gamma2 is %f,
    // %d\n", p1, p2, p3,
    //        gamma, inv_gamma2, dev_params.gravity_on);

    // Scalar p = sqrt(p1 * p1 + p2 * p2 + p3 * p3);
    // Scalar B_sqrt = std::sqrt(tt) / std::abs(q_over_m);
    // Scalar pdotB = (p1 * B1 + p2 * B2 + p3 * B3) / q_over_m;
    // Scalar pitch_angle = pdotB / (p * B_sqrt);

    if (dev_params.rad_cooling_on && sp != (int)ParticleType::ion) {
      sync_kill_perp(p1, p2, p3, gamma, B1, B2, B3, E1, E2, E3,
                     q_over_m);
    }
    // printf("gamma after cooling is %f\n", gamma);
    // printf("p is (%f, %f, %f)\n", p1, p2, p3);
    // printf("pitch angle is %f\n", pitch_angle);
    ptc.p1[idx] = p1;
    ptc.p2[idx] = p2;
    ptc.p3[idx] = p3;
    ptc.E[idx] = gamma;
  }
}
}  // namespace Kernels

}  // namespace Aperture

#endif  // _USER_PUSH_2D_LOGSPH_CUH_
