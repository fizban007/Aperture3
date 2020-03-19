#ifndef _USER_PUSH_2D_LOGSPH_CUH_
#define _USER_PUSH_2D_LOGSPH_CUH_

#include "cuda/algorithms/gravity.cuh"
#include "cuda/algorithms/resonant_cooling.cuh"
#include "cuda/algorithms/sync_cooling.cuh"
#include "cuda/algorithms/vay_push.cuh"
#include "cuda/constant_mem.h"
#include "cuda/data_ptrs.h"

namespace Aperture {

namespace Kernels {

template <int N>
__device__ __forceinline__ void
user_push_2d_logsph(data_ptrs& data, size_t idx, Scalar dt,
                    curandState& state) {
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
  Interpolator2D<Spline::spline_t<N>> interp;
  auto flag = ptc.flag[idx];
  int sp = get_ptc_type(flag);
  auto old_x1 = ptc.x1[idx], old_x2 = ptc.x2[idx];
  auto p1 = ptc.p1[idx], p2 = ptc.p2[idx], p3 = ptc.p3[idx],
       gamma = ptc.E[idx];
  Scalar r = std::exp(dev_mesh.pos(0, c1, old_x1));
  Scalar q_over_m = dt * 0.5f * dev_charges[sp] / dev_masses[sp];
  if (p1 != p1 || p2 != p2 || p3 != p3) {
    printf(
        "NaN detected in push! p1 is %f, p2 is %f, p3 is %f, gamma "
        "is %f\n",
        p1, p2, p3, gamma);
    asm("trap;");
    // p1 = p2 = p3 = 0.0f;
  }
  Scalar E1 =
      interp(data.E1, old_x1, old_x2, c1, c2, Stagger(0b110)) *
      q_over_m;
  Scalar E2 =
      interp(data.E2, old_x1, old_x2, c1, c2, Stagger(0b101)) *
      q_over_m;
  Scalar E3 =
      interp(data.E3, old_x1, old_x2, c1, c2, Stagger(0b011)) *
      q_over_m;
  Scalar B1 =
      interp(data.B1, old_x1, old_x2, c1, c2, Stagger(0b001)) *
      q_over_m;
  Scalar B2 =
      interp(data.B2, old_x1, old_x2, c1, c2, Stagger(0b010)) *
      q_over_m;
  Scalar B3 =
      interp(data.B3, old_x1, old_x2, c1, c2, Stagger(0b100)) *
      q_over_m;

  // step 0: Grab E & M fields at the particle position
  gamma = std::sqrt(1.0f + p1 * p1 + p2 * p2 + p3 * p3);
  if (!check_bit(flag, ParticleFlag::ignore_EM)) {
    vay_push(p1, p2, p3, gamma, E1, E2, E3, B1, B2, B3, q_over_m, dt);
  }

  if (dev_params.gravity_on) {
    gravity(p1, p2, p3, gamma, r, sp, dt);
  }
  // printf("p after is (%f, %f, %f), gamma is %f, inv_gamma2 is %f,
  // %d\n", p1, p2, p3,
  //        gamma, inv_gamma2, dev_params.gravity_on);

  // Scalar p = sqrt(p1 * p1 + p2 * p2 + p3 * p3);
  // Scalar B_sqrt = std::sqrt(tt) / std::abs(q_over_m);
  // Scalar pdotB = (p1 * B1 + p2 * B2 + p3 * B3) / q_over_m;
  // Scalar pitch_angle = pdotB / (p * B_sqrt);
  if (dev_params.rad_cooling_on) {
    sync_kill_perp(p1, p2, p3, gamma, B1, B2, B3, E1, E2, E3,
                   q_over_m);
    // sync_cooling(p1, p2, p3, gamma, B1, B2, B3, E1, E2, E3, q_over_m);
  }

  if (sp != (int)ParticleType::ion) {
    B1 /= q_over_m;
    B2 /= q_over_m;
    B3 /= q_over_m;
    Scalar pdotB = (p1 * B1 + p2 * B2 + p3 * B3);
    Scalar B = sqrt(B1 * B1 + B2 * B2 + B3 * B3);
    Scalar theta = dev_mesh.pos(1, c2, old_x2);
    Scalar gamma_thr_B = dev_params.gamma_thr * B / dev_params.BQ;
    Scalar Eph =
        gamma *
        (1.0f - 1.0f / std::sqrt(1.0f + 2.0f * B / dev_params.BQ));

    // printf("gamma_thr_B is %f, gamma is %f\n",
    //        gamma_thr_B, gamma);
    // if (gamma_thr_B > 3.0f && gamma > gamma_thr_B) {
    // if (Eph > dev_params.E_ph_min && gamma > gamma_thr_B &&
    //     B > 0.5f * dev_params.BQ) {
    if (Eph > dev_params.E_ph_min && gamma > gamma_thr_B) {
      // flag = flag |= bit_or(ParticleFlag::emit_photon);
      ptc.flag[idx] = (flag | bit_or(ParticleFlag::emit_photon));
    }
    // } else if (dev_params.rad_cooling_on) {
    //   // Process resonant drag
    //   Scalar p_mag_signed = sgn(pdotB) * sgn(B1) * std::abs(pdotB) / B;
    //   // printf("p_mag_signed is %f\n", p_mag_signed);
    //   Scalar g = sqrt(1.0f + p_mag_signed * p_mag_signed);
    //   Scalar mu = std::abs(B1 / B);
    //   Scalar y = (B / dev_params.BQ) /
    //              (dev_params.star_kT * (g - p_mag_signed * mu));
    //   // printf("g is %f, y is %f\n", g, y);
    //   if (y < 20.0f && y > 0.0f) {
    //     // printf("y is %f\n", y);
    //     Scalar coef = dev_params.res_drag_coef * y * y * y /
    //                   (r * r * (std::exp(y) - 1.0f));
    //     // printf("coef is %f\n", coef);
    //     // printf("drag coef is %f\n", dev_params.res_drag_coef);
    //     Scalar D = coef * (g * mu - p_mag_signed);
    //     if (B1 < 0.0f) D *= -1.0f;
    //     // printf("D is %f\n", D);
    //     p1 += dt * B1 * D / B;
    //     p2 += dt * B2 * D / B;
    //     p3 += dt * B3 * D / B;
    //     if (p1 != p1 || p2 != p2 || p3 != p3) {
    //         printf(
    //             "NaN detected in resonant cooling! p1 is %f, p2 is %f, p3 is %f, gamma "
    //             "is %f\n",
    //             p1, p2, p3, gamma);
    //         asm("trap;");
    //         // p1 = p2 = p3 = 0.0f;
    //     }
    //     // printf("drag on p1 is %f\n", dt * B1 * D / B);
    //     gamma = sqrt(1.0f + p1 * p1 + p2 * p2 + p3 * p3);

    //     Scalar Ndot = std::abs(coef * (1.0f - p_mag_signed * mu / g));
    //     // printf("gamma after cooling is %f\n", gamma);
    //     // printf("p is (%f, %f, %f)\n", p1, p2, p3);
    //     // printf("pitch angle is %f\n", pitch_angle);
    //     Scalar angle =
    //         acos(sgn(pdotB) * (B1 * cos(theta) - B2 * sin(theta)) / B);
    //     // Scalar theta_p =
    //     //     2.0f * CONST_PI * curand_uniform(&localState);
    //     Scalar theta_p = CONST_PI * curand_uniform(&state);
    //     Scalar phi_p = 2.0f * CONST_PI * curand_uniform(&state);
    //     Scalar u = std::cos(theta_p);
    //     // Scalar beta = sqrt(1.0f - 1.0f / square(g));
    //     // angle = angle + sgn(theta_p - CONST_PI) *
    //     //                     std::acos((u + beta) / (1.0f + beta *
    //     //                     u));
    //     // angle = angle + (2.0f*phi_p - 1.0f) * std::acos((u + beta)
    //     // / (1.0f + beta * u));
    //     Scalar cos_angle =
    //         std::cos(angle) * std::cos(theta_p) +
    //         std::sin(angle) * std::sin(theta_p) * std::cos(phi_p);
    //     angle = std::acos(cos_angle);
    //     auto& ph_flux = data.ph_flux;
    //     Scalar Eph =
    //         (g - std::abs(p_mag_signed) * u) *
    //         (1.0f - 1.0f / sqrt(1.0f + 2.0f * B / dev_params.BQ));
    //     // if (p1 > 0.0f && (Eph < 2.0f || B < 0.1 * dev_params.BQ)) {
    //     if (p1 > 0.0f && gamma > 1.5f && Eph < 2.0f) {
    //       Eph = std::log(std::abs(Eph)) / std::log(10.0f);
    //       if (Eph > 2.0f) Eph = 2.0f;
    //       if (Eph < -6.0f) Eph = -6.0f;
    //       int n0 = ((Eph + 6.0f) / 8.02f * (ph_flux.p.xsize - 1));
    //       if (n0 < 0) n0 = 0;
    //       if (n0 >= ph_flux.p.xsize) n0 = ph_flux.p.xsize - 1;
    //       int n1 = (std::abs(angle) / (CONST_PI + 1.0e-5)) *
    //                (ph_flux.p.ysize - 1);
    //       if (n1 < 0) n1 = 0;
    //       if (n1 >= ph_flux.p.ysize) n1 = ph_flux.p.ysize - 1;
    //       auto w = ptc.weight[idx];
    //       atomicAdd(&ph_flux(n0, n1), Ndot * dt * w);
    //       // printf("n0 is %d, n1 is %d, Ndot is %f, ph_flux is %f\n",
    //       // n0,
    //       //        n1, Ndot, ph_flux(n0, n1));
    //     }
    //   }
    // }
  }
  ptc.p1[idx] = p1;
  ptc.p2[idx] = p2;
  ptc.p3[idx] = p3;
  ptc.E[idx] = gamma;
}

}  // namespace Kernels

}  // namespace Aperture

#endif  // _USER_PUSH_2D_LOGSPH_CUH_
