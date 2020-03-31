#ifndef _USER_PUSH_2D_LOGSPH_CUH_
#define _USER_PUSH_2D_LOGSPH_CUH_

#include "cuda/algorithms/gravity.cu"
#include "cuda/algorithms/ptc_updater_helper.cu"
#include "cuda/algorithms/resonant_cooling.cu"
#include "cuda/algorithms/sync_cooling.cu"
#include "cuda/algorithms/vay_push.cu"
#include "cuda/constant_mem.h"
#include "cuda/data_ptrs.h"
#include "cuda/utils/interpolation.cuh"
#include <curand_kernel.h>

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
  Scalar E1 = interp(data.E1, old_x1, old_x2, c1, c2, Stagger(0b110)) *
              q_over_m;
  Scalar E2 = interp(data.E2, old_x1, old_x2, c1, c2, Stagger(0b101)) *
              q_over_m;
  Scalar E3 = interp(data.E3, old_x1, old_x2, c1, c2, Stagger(0b011)) *
              q_over_m;
  Scalar B1 = interp(data.B1, old_x1, old_x2, c1, c2, Stagger(0b001)) *
              q_over_m;
  Scalar B2 = interp(data.B2, old_x1, old_x2, c1, c2, Stagger(0b010)) *
              q_over_m;
  Scalar B3 = interp(data.B3, old_x1, old_x2, c1, c2, Stagger(0b100)) *
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
    sync_kill_perp(p1, p2, p3, gamma, B1, B2, B3, E1, E2, E3, q_over_m);
    // sync_cooling(p1, p2, p3, gamma, B1, B2, B3, E1, E2, E3,
    // q_over_m);
  }

  if (sp != (int)ParticleType::ion) {
    B1 /= q_over_m;
    B2 /= q_over_m;
    B3 /= q_over_m;
    Scalar p = sqrt(p1 * p1 + p2 * p2 + p3 * p3);
    Scalar B = sqrt(B1 * B1 + B2 * B2 + B3 * B3);
    // B1 /= sgn(pdotB) * B;
    // B2 /= sgn(pdotB) * B;
    // B3 /= sgn(pdotB) * B;
    Scalar pB1 = p1 / p;
    Scalar pB2 = p2 / p;
    Scalar pB3 = p3 / p;
    Scalar theta = dev_mesh.pos(1, c2, old_x2);

    Scalar mu = std::abs(p1 / p);
    // Scalar p_mag_signed = sgn(pdotB) * sgn(B1) * std::abs(pdotB) / B;
    Scalar p_mag_signed = sgn(p1) * p;
    Scalar beta = sqrt(1.0f - 1.0f / (gamma * gamma));
    Scalar y = std::abs((B / dev_params.BQ) /
               (dev_params.star_kT * (gamma - p_mag_signed * mu)));
    if (idx == 0) {
      printf("y is %f, ", y);
    }
    if (y < 20.0f && y > 0.0f) {
      Scalar coef = dev_params.res_drag_coef * square(dev_params.star_kT) * y * y /
                    (r * r * (std::exp(y) - 1.0f));
      Scalar Nph = std::abs(coef / gamma) * dt;
      Scalar Eph = min(
          gamma - 1.0f,
          gamma * (1.0f -
                   1.0f / std::sqrt(1.0f + 2.0f * B / dev_params.BQ)));
      Scalar Eres = (B / dev_params.BQ) / (gamma - p_mag_signed * mu);
      if (idx == 0) {
        printf("Nph is %f, Eph is %f, Eres is %f\n", Nph, Eph, Eres);
        printf("r is %f, theta is %f, gamma is %f, p_par is %f", r, theta, gamma, p_mag_signed);
      }
      // Do not allow the particle to lose too much energy
      if (Eph * Nph > gamma + Eres * Nph - 1.0f) Nph = (gamma + Eres * Nph - 1.0f) / Eph;

      // if (Eph > dev_params.E_ph_min) {
      if (Eph > 2.0f) {
        // Produce individual tracked photons
        if (Nph < 1.0f) {
          float u = curand_uniform(&state);
          if (u < Nph)
            ptc.flag[idx] = (flag | bit_or(ParticleFlag::emit_photon));
        } else {
          ptc.flag[idx] = (flag | bit_or(ParticleFlag::emit_photon));
        }
      } else if (dev_params.rad_cooling_on) {
        // Produce low energy photons that are immediately deposited to
        // an array

        // Draw emission direction in the particle rest frame, z
        // direction is the particle moving direction
        Scalar phi = ptc.x3[idx];
        Scalar theta_p = CONST_PI * curand_uniform(&state);
        Scalar phi_p = 2.0f * CONST_PI * curand_uniform(&state);
        Scalar u = cos(theta_p);
        Scalar cphi = cos(phi_p);
        Scalar sphi = sin(phi_p);

        Eph = gamma * (1.0f + std::abs(beta) * u) *
              (1.0f - 1.0f / sqrt(1.0f + 2.0f * B / dev_params.BQ));

        // Lorentz transform u to the lab frame
        u = (u + beta) / (1 + beta * u);

        Scalar ph1, ph2, ph3;
        ph1 = (pB1 * u - (pB3 * pB3 + pB2 * pB2) * sphi);
        ph2 = (pB2 * u + pB3 * cphi + pB1 * pB2 * sphi);
        ph3 = (pB3 * u - pB2 * cphi + pB1 * pB3 * sphi);
        p1 += Eres * Nph * mu;
        p2 += Eres * Nph * sgn(B1) * sqrt(1.0 - mu * mu);
        p1 -= ph1 * Eph * Nph;
        p2 -= ph2 * Eph * Nph;
        p3 -= ph3 * Eph * Nph;

        auto& ph_flux = data.ph_flux;
        // Compute the theta of the photon outgoing direction
        if (p1 > 0.0f && gamma > 3.5f) {
          logsph2cart(ph1, ph2, ph3, r, theta, phi);
          theta_p = acos(ph3);
          Eph = std::log(std::abs(Eph)) / std::log(10.0f);
          if (Eph > 2.0f) Eph = 2.0f;
          if (Eph < -6.0f) Eph = -6.0f;
          int n0 = ((Eph + 6.0f) / 8.02f * (ph_flux.p.xsize - 1));
          if (n0 < 0) n0 = 0;
          if (n0 >= ph_flux.p.xsize) n0 = ph_flux.p.xsize - 1;
          int n1 = (std::abs(theta_p) / (CONST_PI + 1.0e-5)) *
                   (ph_flux.p.ysize - 1);
          if (n1 < 0) n1 = 0;
          if (n1 >= ph_flux.p.ysize) n1 = ph_flux.p.ysize - 1;
          auto w = ptc.weight[idx];
          atomicAdd(&ph_flux(n0, n1), Nph * w);
          // printf("n0 is %d, n1 is %d, Ndot is %f, ph_flux is %f\n",
          // n0,
          //        n1, Ndot, ph_flux(n0, n1));
        }
      }
    }

    if (idx == 0) {
      printf("\n");
    }
  }
  ptc.p1[idx] = p1;
  ptc.p2[idx] = p2;
  ptc.p3[idx] = p3;
  ptc.E[idx] = sqrt(1.0 + p1 * p1 + p2 * p2 + p3 * p3);
}

}  // namespace Kernels

}  // namespace Aperture

#endif  // _USER_PUSH_2D_LOGSPH_CUH_
