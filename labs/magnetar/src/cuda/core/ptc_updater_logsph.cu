#include "core/detail/multi_array_utils.hpp"
#include "cuda/constant_mem.h"
#include "cuda/core/cu_sim_data.h"
#include "cuda/core/ptc_updater_helper.cuh"
#include "cuda/core/ptc_updater_logsph.h"
#include "cuda/core/sim_environment_dev.h"
#include "cuda/cudaUtility.h"
#include "cuda/kernels.h"
#include "cuda/ptr_util.h"
#include "cuda/utils/interpolation.cuh"
#include "utils/logger.h"
#include "utils/util_functions.h"

#define DEPOSIT_EPS 1.0e-10f

namespace Aperture {

namespace Kernels {

HD_INLINE void
cart2logsph(Scalar &v1, Scalar &v2, Scalar &v3, Scalar x1, Scalar x2,
            Scalar x3) {
  Scalar v1n = v1, v2n = v2, v3n = v3;
  v1 =
      v1n * sin(x2) * cos(x3) + v2n * sin(x2) * sin(x3) + v3n * cos(x2);
  v2 =
      v1n * cos(x2) * cos(x3) + v2n * cos(x2) * sin(x3) - v3n * sin(x2);
  v3 = -v1n * sin(x3) + v2n * cos(x3);
}

HD_INLINE void
logsph2cart(Scalar &v1, Scalar &v2, Scalar &v3, Scalar x1, Scalar x2,
            Scalar x3) {
  Scalar v1n = v1, v2n = v2, v3n = v3;
  v1 =
      v1n * sin(x2) * cos(x3) + v2n * cos(x2) * cos(x3) - v3n * sin(x3);
  v2 =
      v1n * sin(x2) * sin(x3) + v2n * cos(x2) * sin(x3) + v3n * cos(x3);
  v3 = v1n * cos(x2) - v2n * sin(x2);
}

__global__ void
vay_push_2d(particle_data ptc, size_t num,
            PtcUpdaterDev::fields_data fields, cudaPitchedPtr ph_flux,
            curandState *states, Scalar dt) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  curandState localState = states[id];
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num;
       idx += blockDim.x * gridDim.x) {
    auto c = ptc.cell[idx];
    // Skip empty particles
    if (c == MAX_CELL || idx >= num) continue;

    // Load particle quantities
    Interpolator2D<spline_t> interp;
    auto flag = ptc.flag[idx];
    int sp = get_ptc_type(flag);
    auto old_x1 = ptc.x1[idx], old_x2 = ptc.x2[idx];
    auto p1 = ptc.p1[idx], p2 = ptc.p2[idx], p3 = ptc.p3[idx];
    int c1 = dev_mesh.get_c1(c);
    int c2 = dev_mesh.get_c2(c);
    Scalar q_over_m = dt * 0.5f * dev_charges[sp] / dev_masses[sp];
    // step 0: Grab E & M fields at the particle position
    Scalar gamma = std::sqrt(1.0f + p1 * p1 + p2 * p2 + p3 * p3);
    if (!check_bit(flag, ParticleFlag::ignore_EM)) {
      Scalar E1 =
          (interp(fields.E1, old_x1, old_x2, c1, c2, Stagger(0b110))) *
          // interp(dev_bg_fields.E1, old_x1, old_x2, c1, c2,
          //        Stagger(0b110))) *
          q_over_m;
      Scalar E2 =
          (interp(fields.E2, old_x1, old_x2, c1, c2, Stagger(0b101))) *
          // interp(dev_bg_fields.E2, old_x1, old_x2, c1, c2,
          //        Stagger(0b101))) *
          q_over_m;
      Scalar E3 =
          (interp(fields.E3, old_x1, old_x2, c1, c2, Stagger(0b011))) *
          // interp(dev_bg_fields.E3, old_x1, old_x2, c1, c2,
          //        Stagger(0b011))) *
          q_over_m;
      Scalar B1 =
          (interp(fields.B1, old_x1, old_x2, c1, c2, Stagger(0b001)) +
           interp(dev_bg_fields.B1, old_x1, old_x2, c1, c2,
                  Stagger(0b001))) *
          q_over_m;
      Scalar B2 =
          (interp(fields.B2, old_x1, old_x2, c1, c2, Stagger(0b010)) +
           interp(dev_bg_fields.B2, old_x1, old_x2, c1, c2,
                  Stagger(0b010))) *
          q_over_m;
      Scalar B3 =
          (interp(fields.B3, old_x1, old_x2, c1, c2, Stagger(0b100))) *
          // interp(dev_bg_fields.B3, old_x1, old_x2, c1, c2,
          //        Stagger(0b100))) *
          q_over_m;

      // printf("B1 = %f, B2 = %f, B3 = %f\n", B1, B2, B3);
      // printf("E1 = %f, E2 = %f, E3 = %f\n", E1, E2, E3);
      // printf("B cell is %f\n", *ptrAddr(fields.B1, c1*sizeof(Scalar)
      // + c2*fields.B1.pitch)); printf("q over m is %f\n", q_over_m);
      // printf("gamma before is %f\n", gamma);
      // printf("p is (%f, %f, %f), gamma is %f\n", p1, p2, p3, gamma);

      // step 1: Update particle momentum using vay pusher
      Scalar up1 = p1 + 2.0f * E1 + (p2 * B3 - p3 * B2) / gamma;
      Scalar up2 = p2 + 2.0f * E2 + (p3 * B1 - p1 * B3) / gamma;
      Scalar up3 = p3 + 2.0f * E3 + (p1 * B2 - p2 * B1) / gamma;
      // printf("p prime is (%f, %f, %f), gamma is %f\n", up1, up2, up3,
      // gamma);
      Scalar tt = B1 * B1 + B2 * B2 + B3 * B3;
      Scalar ut = up1 * B1 + up2 * B2 + up3 * B3;

      Scalar sigma = 1.0f + up1 * up1 + up2 * up2 + up3 * up3 - tt;
      Scalar inv_gamma2 =
          2.0f /
          (sigma + std::sqrt(sigma * sigma + 4.0f * (tt + ut * ut)));
      Scalar s = 1.0f / (1.0f + inv_gamma2 * tt);
      gamma = 1.0f / std::sqrt(inv_gamma2);

      p1 =
          (up1 + B1 * ut * inv_gamma2 + (up2 * B3 - up3 * B2) / gamma) *
          s;
      p2 =
          (up2 + B2 * ut * inv_gamma2 + (up3 * B1 - up1 * B3) / gamma) *
          s;
      p3 =
          (up3 + B3 * ut * inv_gamma2 + (up1 * B2 - up2 * B1) / gamma) *
          s;

      // printf("p after is (%f, %f, %f), gamma is %f, inv_gamma2 is %f,
      // %d\n", p1, p2, p3,
      //        gamma, inv_gamma2, dev_params.gravity_on);
      // Add an artificial gravity
      Scalar r = exp(dev_mesh.pos(0, c1, old_x1));
      Scalar theta = dev_mesh.pos(1, c2, old_x2);
      if (dev_params.gravity_on) {
        p1 -= dt * dev_params.gravity / (r * r);
        gamma = sqrt(1.0f + p1 * p1 + p2 * p2 + p3 * p3);
      }

      Scalar pdotB = p1 * B1 + p2 * B2 + p3 * B3;
      if (dev_params.rad_cooling_on && sp != (int)ParticleType::ion) {
        Scalar pp1 = p1 - B1 * pdotB / tt;
        Scalar pp2 = p2 - B2 * pdotB / tt;
        Scalar pp3 = p3 - B3 * pdotB / tt;
        Scalar pp = sqrt(pp1 * pp1 + pp2 * pp2 + pp3 * pp3);
        // Scalar p = sqrt(p1 * p1 + p2 * p2 + p3 * p3);
        // printf("pp * dt is %f\n", pp * dt);

        p1 -= pp1 * dt * dev_params.rad_cooling_coef;
        p2 -= pp2 * dt * dev_params.rad_cooling_coef;
        p3 -= pp3 * dt * dev_params.rad_cooling_coef;
        // dev_params.rad_cooling_coef;
        // p3 -= pp3 * (2.0f * dt * pp * tt / 3.0f) *
        //       dev_params.rad_cooling_coef;
      }
      Scalar B = sqrt(tt) / std::abs(q_over_m);
      B1 /= q_over_m;
      B2 /= q_over_m;
      B3 /= q_over_m;
      Scalar gamma_thr_B = dev_params.gamma_thr * B / dev_params.BQ;
      // printf("B is %f\n", B);
      if (gamma_thr_B > 3.0f && gamma > gamma_thr_B &&
          sp != (int)ParticleType::ion) {
        ptc.flag[idx] = flag |= bit_or(ParticleFlag::emit_photon);
      } else if (dev_params.rad_cooling_on &&
                 sp != (int)ParticleType::ion) {
        // Process resonant drag
        // Scalar p_mag = std::abs(pdotB / B);
        Scalar p_mag_signed = sgn(pdotB / q_over_m) * sgn(B1) *
                              std::abs(pdotB / q_over_m) / B;
        // printf("p_mag_signed is %f\n", p_mag_signed);
        Scalar g = sqrt(1.0f + p_mag_signed * p_mag_signed);
        Scalar mu = std::abs(B1 / B);
        Scalar y = (B / dev_params.BQ) /
                   (dev_params.star_kT * (g - p_mag_signed * mu));
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

          // Scalar Ndot = std::abs(coef * (1.0f - p_mag_signed * mu /
          // g)); Scalar angle =
          //     acos(sgn(pdotB) * (B1 * cos(theta) - B2 * sin(theta)) /
          //     B);
          // Scalar theta_p = 2.0f * CONST_PI *
          // curand_uniform(&localState); Scalar u = std::cos(theta_p);
          // Scalar beta = sqrt(1.0f - 1.0f / square(g));
          // angle = angle + sgn(theta_p - CONST_PI) *
          //                     std::acos((u + beta) / (1.0f + beta *
          //                     u));
          // angle = std::acos(std::cos(angle));
          // Scalar Eph =
          //     (g - std::abs(p_mag_signed) * u) *
          //     (1.0f - 1.0f / sqrt(1.0f + 2.0f * B / dev_params.BQ));
          // Eph = std::log(Eph) / std::log(10.0f);
          // if (Eph > 0.0f) Eph = 0.0f;
          // if (Eph < -9.0f) Eph = -9.0f;
          // int n0 = ((Eph + 9.0f) / 9.0f *
          //           (ph_flux.xsize / sizeof(float) - 1));
          // int n1 = (std::abs(angle) / CONST_PI) * (ph_flux.ysize -
          // 1); atomicAdd(ptrAddr(ph_flux, n0, n1), Ndot * dt);
        }
      }
      ptc.p1[idx] = p1;
      ptc.p2[idx] = p2;
      ptc.p3[idx] = p3;
      ptc.E[idx] = gamma;
      // printf("gamma is %f\n", gamma);
    }
  }
  states[id] = localState;
}

__global__ void
move_photons(photon_data photons, size_t num, Scalar dt) {
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num;
       idx += blockDim.x * gridDim.x) {
    auto c = photons.cell[idx];
    // Skip empty particles
    if (c == MAX_CELL) continue;
    // Load particle quantities
    int c1 = dev_mesh.get_c1(c);
    int c2 = dev_mesh.get_c2(c);
    auto v1 = photons.p1[idx], v2 = photons.p2[idx],
         v3 = photons.p3[idx];
    Scalar E = std::sqrt(v1 * v1 + v2 * v2 + v3 * v3);
    v1 = v1 / E;
    v2 = v2 / E;
    v3 = v3 / E;

    auto old_x1 = photons.x1[idx], old_x2 = photons.x2[idx],
         old_x3 = photons.x3[idx];

    // Compute the actual movement
    Scalar r1 = dev_mesh.pos(0, c1, old_x1);
    Scalar exp_r1 = std::exp(r1);

    // Censor photons already outside the conversion radius
    if (exp_r1 > dev_params.r_cutoff || exp_r1 < 1.02) {
      photons.cell[idx] = MAX_CELL;
      continue;
    }

    Scalar r2 = dev_mesh.pos(1, c2, old_x2);
    Scalar x = exp_r1 * std::sin(r2) * std::cos(old_x3);
    Scalar y = exp_r1 * std::sin(r2) * std::sin(old_x3);
    Scalar z = exp_r1 * std::cos(r2);

    logsph2cart(v1, v2, v3, r1, r2, old_x3);
    x += v1 * dt;
    y += v2 * dt;
    z += v3 * dt;
    Scalar r1p = sqrt(x * x + y * y + z * z);
    Scalar r2p = acos(z / r1p);
    r1p = log(r1p);
    Scalar r3p = atan(y / x);
    if (x < 0.0f) v1 *= -1.0f;

    cart2logsph(v1, v2, v3, r1p, r2p, r3p);
    photons.p1[idx] = v1 * E;
    photons.p2[idx] = v2 * E;
    photons.p3[idx] = v3 * E;

    Pos_t new_x1 = old_x1 + (r1p - r1) / dev_mesh.delta[0];
    Pos_t new_x2 = old_x2 + (r2p - r2) / dev_mesh.delta[1];
    // printf("new_x1 is %f, new_x2 is %f, old_x1 is %f, old_x2 is
    // %f\n", new_x1, new_x2, old_x1, old_x2);
    int dc1 = floor(new_x1);
    int dc2 = floor(new_x2);
    photons.cell[idx] = dev_mesh.get_idx(c1 + dc1, c2 + dc2);
    new_x1 -= (Pos_t)dc1;
    new_x2 -= (Pos_t)dc2;
    // printf("new_x1 is %f, new_x2 is %f, dc2 = %d\n", new_x1, new_x2,
    // dc2);
    photons.x1[idx] = new_x1;
    photons.x2[idx] = new_x2;
    photons.x3[idx] = r3p;
    photons.path_left[idx] -= dt;
  }
}

__global__ void
__launch_bounds__(512, 4)
    deposit_current_2d_log_sph(particle_data ptc, size_t num,
                               PtcUpdaterDev::fields_data fields,
                               Grid_LogSph_dev::mesh_ptrs mesh_ptrs,
                               Scalar dt) {
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num;
       idx += blockDim.x * gridDim.x) {
    auto c = ptc.cell[idx];
    // Skip empty particles
    if (c == MAX_CELL || idx >= num) continue;

    // Load particle quantities
    Interpolator2D<spline_t> interp;
    int c1 = dev_mesh.get_c1(c);
    int c2 = dev_mesh.get_c2(c);
    auto v1 = ptc.p1[idx], v2 = ptc.p2[idx], v3 = ptc.p3[idx];
    Scalar gamma = ptc.E[idx];
    // printf("gamma is %f\n", gamma);
    // printf("velocity before is (%f, %f, %f)\n", v1, v2, v3);

    auto flag = ptc.flag[idx];
    int sp = get_ptc_type(flag);
    auto w = ptc.weight[idx];
    auto old_x1 = ptc.x1[idx], old_x2 = ptc.x2[idx],
         old_x3 = ptc.x3[idx];

    v1 = v1 / gamma;
    v2 = v2 / gamma;
    v3 = v3 / gamma;

    // step 1: Compute particle movement and update position
    Scalar r1 = dev_mesh.pos(0, c1, old_x1);
    Scalar exp_r1 = std::exp(r1);
    Scalar r2 = dev_mesh.pos(1, c2, old_x2);
    Scalar x = exp_r1 * std::sin(r2) * std::cos(old_x3);
    Scalar y = exp_r1 * std::sin(r2) * std::sin(old_x3);
    Scalar z = exp_r1 * std::cos(r2);
    // printf("cart position is (%f, %f, %f)\n", x, y, z);

    logsph2cart(v1, v2, v3, r1, r2, old_x3);
    // printf("cart velocity is (%f, %f, %f)\n", v1, v2, v3);
    x += v1 * dt;
    y += v2 * dt;
    z += v3 * dt;
    // printf("new cart position is (%f, %f, %f)\n", x, y, z);
    Scalar r1p = sqrt(x * x + y * y + z * z);
    Scalar r2p = acos(z / r1p);
    r1p = log(r1p);
    Scalar r3p = atan(y / x);
    if (x < 0.0f) v1 *= -1.0f;

    // printf("new position is (%f, %f, %f)\n", exp(r1p), r2p, r3p);

    cart2logsph(v1, v2, v3, r1p, r2p, r3p);
    ptc.p1[idx] = v1 * gamma;
    ptc.p2[idx] = v2 * gamma;
    ptc.p3[idx] = v3 * gamma;

    // Scalar old_pos3 =
    Pos_t new_x1 = old_x1 + (r1p - r1) / dev_mesh.delta[0];
    Pos_t new_x2 = old_x2 + (r2p - r2) / dev_mesh.delta[1];
    // printf("new_x1 is %f, new_x2 is %f, old_x1 is %f, old_x2 is
    // %f\n", new_x1, new_x2, old_x1, old_x2);
    int dc1 = floor(new_x1);
    int dc2 = floor(new_x2);
#ifndef NDEBUG
    if (dc1 > 1 || dc1 < -1 || dc2 > 1 || dc2 < -1)
      printf("----------------- Error: moved more than 1 cell!");
#endif
    new_x1 -= (Pos_t)dc1;
    new_x2 -= (Pos_t)dc2;
    // reflect around the axis
    if (c2 + dc2 < dev_mesh.guard[1]) {
      dc2 += 1;
      new_x2 = 1.0f - new_x2;
    } else if (c2 + dc2 >= dev_mesh.dims[1] - dev_mesh.guard[1]) {
      dc2 -= 1;
      new_x2 = 1.0f - new_x2;
    }
    ptc.cell[idx] = dev_mesh.get_idx(c1 + dc1, c2 + dc2);
    // printf("new_x1 is %f, new_x2 is %f, dc2 = %d\n", new_x1, new_x2,
    // dc2);
    ptc.x1[idx] = new_x1;
    ptc.x2[idx] = new_x2;
    ptc.x3[idx] = r3p;

    // step 2: Deposit current
    if (check_bit(flag, ParticleFlag::ignore_current)) continue;
    // Scalar djz[spline_t::support + 1][spline_t::support + 1] =
    // {0.0f};
    Scalar weight = -dev_charges[sp] * w;

    int j_0 = (dc2 == -1 ? -2 : -1);
    int j_1 = (dc2 == 1 ? 1 : 0);
    int i_0 = (dc1 == -1 ? -2 : -1);
    int i_1 = (dc1 == 1 ? 1 : 0);
    Scalar djy[3] = {0.0f};
    for (int j = j_0; j <= j_1; j++) {
      Scalar sy0 = interp.interpolate(-old_x2 + j + 1);
      Scalar sy1 = interp.interpolate(-new_x2 + (j + 1 - dc2));

      size_t j_offset = (j + c2) * fields.J1.pitch;
      Scalar djx = 0.0f;
      for (int i = i_0; i <= i_1; i++) {
        Scalar sx0 = interp.interpolate(-old_x1 + i + 1);
        Scalar sx1 = interp.interpolate(-new_x1 + (i + 1 - dc1));

        // j1 is movement in r
        int offset = j_offset + (i + c1) * sizeof(Scalar);
        Scalar val0 = movement2d(sy0, sy1, sx0, sx1);
        djx += val0;
        atomicAdd(ptrAddr(fields.J1, offset + sizeof(Scalar)),
                  weight * djx);

        // j2 is movement in theta
        Scalar val1 = movement2d(sx0, sx1, sy0, sy1);
        djy[i - i_0] += val1;
        atomicAdd(ptrAddr(fields.J2, offset + fields.J2.pitch),
                  weight * djy[i - i_0]);

        // j3 is simply v3 times rho at volume average
        Scalar val2 = center2d(sx0, sx1, sy0, sy1);
        atomicAdd(ptrAddr(fields.J3, offset),
                  -weight * v3 * val2 / *ptrAddr(mesh_ptrs.dV, offset));

        // rho is deposited at the final position
        Scalar s1 = sx1 * sy1;
        atomicAdd(ptrAddr(fields.Rho[sp], offset), -weight * s1);
      }
    }
  }
}

__global__ void
convert_j(cudaPitchedPtr j1, cudaPitchedPtr j2,
          PtcUpdaterDev::fields_data fields) {
  for (int j = blockIdx.y * blockDim.y + threadIdx.y;
       j < dev_mesh.dims[1]; j += blockDim.y * gridDim.y) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < dev_mesh.dims[0]; i += blockDim.x * gridDim.x) {
      size_t offset_f = j * fields.J1.pitch + i * sizeof(Scalar);
      size_t offset_d = j * j1.pitch + i * sizeof(double);
      (*ptrAddr(fields.J1, offset_f)) =
          (*(float2 *)((char *)j1.ptr + offset_d)).x;
      (*ptrAddr(fields.J2, offset_f)) =
          (*(float2 *)((char *)j2.ptr + offset_d)).x;
    }
  }
}

__global__ void
process_j(PtcUpdaterDev::fields_data fields,
          Grid_LogSph_dev::mesh_ptrs mesh_ptrs, Scalar dt) {
  for (int j = blockIdx.y * blockDim.y + threadIdx.y;
       j < dev_mesh.dims[1]; j += blockDim.y * gridDim.y) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < dev_mesh.dims[0]; i += blockDim.x * gridDim.x) {
      size_t offset = j * fields.J1.pitch + i * sizeof(Scalar);
      Scalar w = dev_mesh.delta[0] * dev_mesh.delta[1] / dt;
      (*ptrAddr(fields.J1, offset)) *=
          w / *ptrAddr(mesh_ptrs.A1_e, offset);
      (*ptrAddr(fields.J2, offset)) *=
          w / *ptrAddr(mesh_ptrs.A2_e, offset);
      for (int n = 0; n < dev_params.num_species; n++) {
        (*ptrAddr(fields.Rho[n], offset)) /=
            *ptrAddr(mesh_ptrs.dV, offset);
      }
    }
  }
}

__global__ void
inject_ptc(particle_data ptc, size_t num, int inj_per_cell, Scalar p1,
           Scalar p2, Scalar p3, Scalar w, curandState *states,
           Scalar omega) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  curandState localState = states[id];
  for (int i = dev_mesh.guard[1] + 1 + id;
       // i = dev_mesh.dims[1] - dev_mesh.guard[1] - 3 + id;
       i < dev_mesh.dims[1] - dev_mesh.guard[1] - 1;
       i += blockDim.x * gridDim.x) {
    size_t offset = num + i * inj_per_cell * 2;
    Scalar r = exp(dev_mesh.pos(0, dev_mesh.guard[0] + 2, 0.5f));
    for (int n = 0; n < inj_per_cell; n++) {
      Pos_t x2 = curand_uniform(&localState);
      Scalar theta = dev_mesh.pos(1, i, x2);
      // Scalar vphi = omega * r * cos(theta) * sin(theta);
      Scalar vphi = 0.0f;
      ptc.x1[offset + n * 2] = 0.5f;
      ptc.x2[offset + n * 2] = x2;
      ptc.x3[offset + n * 2] = 0.0f;
      ptc.p1[offset + n * 2] = p1;
      ptc.p2[offset + n * 2] = p2;
      ptc.p3[offset + n * 2] = vphi;
      ptc.E[offset + n * 2] =
          sqrt(1.0f + p1 * p1 + p2 * p2 + vphi * vphi);
      // printf("inject E is %f\n", ptc.E[offset + n * 2]);
      // ptc.p3[offset + n * 2] = p3;
      ptc.cell[offset + n * 2] =
          dev_mesh.get_idx(dev_mesh.guard[0] + 2, i);
      ptc.weight[offset + n * 2] =
          w * sin(theta) * std::abs(cos(theta));
      ptc.flag[offset + n * 2] = set_ptc_type_flag(
          bit_or(ParticleFlag::primary), ParticleType::electron);

      ptc.x1[offset + n * 2 + 1] = 0.5f;
      ptc.x2[offset + n * 2 + 1] = x2;
      ptc.x3[offset + n * 2 + 1] = 0.0f;
      ptc.p1[offset + n * 2 + 1] = p1;
      ptc.p2[offset + n * 2 + 1] = p2;
      ptc.p3[offset + n * 2 + 1] = vphi;
      ptc.E[offset + n * 2 + 1] =
          sqrt(1.0f + p1 * p1 + p2 * p2 + vphi * vphi);
      // printf("inject E is %f\n", ptc.E[offset + n * 2 + 1]);
      // ptc.p3[offset + n * 2 + 1] = p3;
      ptc.cell[offset + n * 2 + 1] =
          dev_mesh.get_idx(dev_mesh.guard[0] + 2, i);
      ptc.weight[offset + n * 2 + 1] =
          w * sin(theta) * std::abs(cos(theta));
      ptc.flag[offset + n * 2 + 1] = set_ptc_type_flag(
          bit_or(ParticleFlag::primary), ParticleType::ion);
    }
  }
  states[id] = localState;
}

__global__ void
boundary_rho(PtcUpdaterDev::fields_data fields,
             Grid_LogSph_dev::mesh_ptrs mesh_ptrs) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < dev_mesh.dims[0]; i += blockDim.x * gridDim.x) {
    size_t offset_0 =
        i * sizeof(Scalar) + dev_mesh.guard[1] * fields.Rho[0].pitch;
    size_t offset_pi = i * sizeof(Scalar) +
                       (dev_mesh.dims[1] - dev_mesh.guard[1] - 2) *
                           fields.Rho[0].pitch;
    for (int n = 0; n < dev_params.num_species; n++) {
      // (*ptrAddr(fields.Rho[n], offset_0)) +=
      //     *ptrAddr(fields.Rho[n], offset_0 - 2 * fields.Rho[n].pitch)
      //     * *ptrAddr(mesh_ptrs.dV, offset_0 - 2 *
      //     fields.Rho[n].pitch) / *ptrAddr(mesh_ptrs.dV, offset_0);
      // (*ptrAddr(fields.Rho[n], offset_pi)) +=
      //     *ptrAddr(fields.Rho[n], offset_pi + 2 *
      //     fields.Rho[n].pitch) * *ptrAddr(mesh_ptrs.dV, offset_pi + 2
      //     * fields.Rho[n].pitch) / *ptrAddr(mesh_ptrs.dV, offset_pi);

      // (*ptrAddr(fields.Rho[n], offset_0 - 2 * fields.Rho[0].pitch)) =
      //     0.0f;
      // (*ptrAddr(fields.Rho[n], offset_pi + 2 * fields.Rho[0].pitch))
      // =
      //     0.0f;
    }
    // (*ptrAddr(fields.J1, offset_0)) -=
    //     *ptrAddr(fields.J1, offset_0 - 2 * fields.J1.pitch);
    // (*ptrAddr(fields.J1, offset_pi)) +=
    //     *ptrAddr(fields.J1, offset_pi + 2 * fields.J1.pitch) *
    //     *ptrAddr(mesh_ptrs.A1_e, offset_pi + 2 * fields.J1.pitch) /
    //     *ptrAddr(mesh_ptrs.A1_e, offset_pi);

    // *ptrAddr(fields.J1, offset_0 - 2 * fields.J1.pitch) = 0.0f;
    // *ptrAddr(fields.J1, offset_pi + 2 * fields.J1.pitch) = 0.0f;

    // (*ptrAddr(fields.J2, offset_0)) -=
    //     *ptrAddr(fields.J2, offset_0 - fields.J2.pitch);
    (*ptrAddr(fields.J2, offset_pi + fields.J2.pitch)) -=
        *ptrAddr(fields.J2, offset_pi + 2 * fields.J2.pitch);

    (*ptrAddr(fields.J3, offset_0 - fields.J3.pitch)) = 0.0f;
    (*ptrAddr(fields.J3, offset_pi + fields.J3.pitch)) = 0.0f;
    // (*ptrAddr(fields.J2, offset_0 - fields.J2.pitch)) = 0.0f;
    // (*ptrAddr(fields.J2, offset_pi)) = 0.0f;
    // (*ptrAddr(fields.J2, offset_pi - fields.J2.pitch)) -=
    //     *ptrAddr(fields.J2, offset_pi + fields.J2.pitch);
  }
}

__global__ void
annihilate_pairs(particle_data ptc, size_t num, cudaPitchedPtr j1,
                 cudaPitchedPtr j2, cudaPitchedPtr j3,
                 Grid_LogSph_dev::mesh_ptrs mesh_ptrs, Scalar dt) {
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num;
       idx += blockDim.x * gridDim.x) {
    // First do a deposit before annihilation
    auto c = ptc.cell[idx];
    auto flag = ptc.flag[idx];
    // Skip empty particles
    if (c == MAX_CELL) continue;

    if (check_bit(flag, ParticleFlag::annihilate)) {
      // Load particle quantities
      Interpolator2D<spline_t> interp;
      int c1 = dev_mesh.get_c1(c);
      int c2 = dev_mesh.get_c2(c);
      int sp = get_ptc_type(flag);
      auto w = ptc.weight[idx];
      auto old_x1 = ptc.x1[idx], old_x2 = ptc.x2[idx];

      Pos_t new_x1 = 0.5f;
      Pos_t new_x2 = 0.5f;

      // Move the particles to be annihilated to the center of the cell
      ptc.x1[idx] = new_x1;
      ptc.x2[idx] = new_x2;

      // Deposit extra current due to this movement
      if (!check_bit(flag, ParticleFlag::ignore_current)) {
        // Scalar djz[spline_t::support + 1][spline_t::support + 1] =
        // {0.0f};
        Scalar weight = -dev_charges[sp] * w;

        Scalar djy[3] = {0.0f};
        for (int j = -1; j <= 0; j++) {
          Scalar sy0 = interp.interpolate(-old_x2 + j + 1);
          Scalar sy1 = interp.interpolate(-new_x2 + j + 1);

          size_t j_offset = (j + c2) * j1.pitch;
          Scalar djx = 0.0f;
          for (int i = -1; i <= 0; i++) {
            Scalar sx0 = interp.interpolate(-old_x1 + i + 1);
            Scalar sx1 = interp.interpolate(-new_x1 + i + 1);

            // j1 is movement in r
            int offset = j_offset + (i + c1) * sizeof(Scalar);
            Scalar val0 = movement2d(sy0, sy1, sx0, sx1);
            djx += val0;
            atomicAdd(ptrAddr(j1, offset + sizeof(Scalar)),
                      weight * djx * dev_mesh.delta[0] *
                          dev_mesh.delta[1] /
                          (dt * *ptrAddr(mesh_ptrs.A1_e,
                                         offset + sizeof(Scalar))));

            // j2 is movement in theta
            Scalar val1 = movement2d(sx0, sx1, sy0, sy1);
            djy[i + 1] += val1;
            atomicAdd(
                ptrAddr(j2, offset + j2.pitch),
                weight * djy[i + 1] * dev_mesh.delta[0] *
                    dev_mesh.delta[1] /
                    (dt * *ptrAddr(mesh_ptrs.A2_e, offset + j2.pitch)));
          }
        }
      }

      // Actually kill the particle
      ptc.cell[idx] = MAX_CELL;
      ptc.flag[idx] = 0;
    }
  }
}

__global__ void
flag_annihilation(particle_data ptc, size_t num, cudaPitchedPtr dens_e,
                  cudaPitchedPtr dens_p, cudaPitchedPtr balance,
                  cudaPitchedPtr annihilate) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num;
       i += blockDim.x * gridDim.x) {
    auto c = ptc.cell[i];
    // Skip empty particles
    if (c == MAX_CELL) continue;

    auto flag = ptc.flag[i];
    if (get_ptc_type(flag) > 1) continue;  // ignore ions
    auto w = ptc.weight[i];

    int c1 = dev_mesh.get_c1(c);
    int c2 = dev_mesh.get_c2(c);
    Scalar sin_t = std::sin(dev_mesh.pos(1, c2, 0.5f));
    int ann = *(int *)((char *)annihilate.ptr + c1 * sizeof(int) +
                       c2 * annihilate.pitch);
    Scalar n_min = 0.2 * square(dev_mesh.inv_delta[0]) * sin_t;
    Scalar n;
    if (ann != 0) {
      if (get_ptc_type(flag) == (int)ParticleType::electron)
        n = atomicAdd(ptrAddr(dens_e, c1, c2), w);
      else  // if (get_ptc_type(flag) == (int)ParticleType::positron)
        n = atomicAdd(ptrAddr(dens_p, c1, c2), w);
      if (n < n_min) {
        set_bit(ptc.flag[i], ParticleFlag::annihilate);
        atomicAdd(ptrAddr(balance, c1, c2),
                  w * (get_ptc_type(flag) == (int)ParticleType::electron
                           ? -1.0f
                           : 1.0f));
      }
    }
    // Scalar sin_t = std::sin(dev_mesh.pos(1, c2, 0.5f));
    // // size_t offset = c1 * sizeof(Scalar) + c2 * dens.pitch;

    // Scalar n_e = *ptrAddr(dens_e, c1, c2);
    // Scalar n_p = *ptrAddr(dens_p, c1, c2);
    // if (get_ptc_type(flag) == (int)ParticleType::electron)
    //   n_e = atomicAdd(ptrAddr(dens_e, c1, c2), w);
    // else if (get_ptc_type(flag) == (int)ParticleType::positron)
    //   n_p = atomicAdd(ptrAddr(dens_p, c1, c2), w);
    // Scalar r = std::exp(dev_mesh.pos(0, c1, 0.5f));
    // Scalar n_min = 0.2 * square(dev_mesh.inv_delta[0]) * sin_t;
    // // TODO: implement the proper condition
    // if (n_e > n_min && n_p > n_min) {
    //   set_bit(ptc.flag[i], ParticleFlag::annihilate);
    //   atomicAdd(ptrAddr(balance, c1, c2),
    //             w * (get_ptc_type(flag) ==
    //             (int)ParticleType::electron
    //                      ? -1.0f
    //                      : 1.0f));
    // }
  }
  // After this operation, the balance array will contain how much
  // imbalance is there in the annihilated part. We will add this
  // imbalance back in as an extra particle
}

__global__ void
check_annihilation(particle_data ptc, size_t num, cudaPitchedPtr dens_e,
                   cudaPitchedPtr dens_p, cudaPitchedPtr annihilate) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num;
       i += blockDim.x * gridDim.x) {
    auto c = ptc.cell[i];
    // Skip empty particles
    if (c == MAX_CELL) continue;

    auto flag = ptc.flag[i];
    if (get_ptc_type(flag) > 1) continue;  // ignore ions
    auto w = ptc.weight[i];

    int c1 = dev_mesh.get_c1(c);
    int c2 = dev_mesh.get_c2(c);
    size_t offset = c1 * sizeof(int) + c2 * annihilate.pitch;
    int b = *(int *)((char *)annihilate.ptr + offset);
    Scalar sin_t = std::sin(dev_mesh.pos(1, c2, 0.5f));
    // size_t offset = c1 * sizeof(Scalar) + c2 * dens.pitch;

    Scalar n_e = *ptrAddr(dens_e, c1, c2);
    Scalar n_p = *ptrAddr(dens_p, c1, c2);
    if (get_ptc_type(flag) == (int)ParticleType::electron)
      n_e = atomicAdd(ptrAddr(dens_e, c1, c2), w);
    else if (get_ptc_type(flag) == (int)ParticleType::positron)
      n_p = atomicAdd(ptrAddr(dens_p, c1, c2), w);
    Scalar n_min = 0.2 * square(dev_mesh.inv_delta[0]) * sin_t;

    if (n_e > n_min && n_p > n_min && b == 0) {
      atomicExch((int *)((char *)annihilate.ptr + offset), 1);
    }
  }
}

// __global__ void
// annihilate_pairs(particle_data ptc, size_t num, cudaPitchedPtr j1,
//                  cudaPitchedPtr j2, cudaPitchedPtr j3) {
//   for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num;
//        idx += blockDim.x * gridDim.x) {
//     // First do a deposit before annihilation
//     auto c = ptc.cell[idx];
//     auto flag = ptc.flag[idx];
//     // Skip empty particles
//     if (c == MAX_CELL || !check_bit(flag, ParticleFlag::annihilate))
//       continue;

//     // Load particle quantities
//     Interpolator2D<spline_t> interp;
//     int c1 = dev_mesh.get_c1(c);
//     int c2 = dev_mesh.get_c2(c);
//     int sp = get_ptc_type(flag);
//     auto w = ptc.weight[idx];
//     auto old_x1 = ptc.x1[idx], old_x2 = ptc.x2[idx];

//     Pos_t new_x1 = 0.5f;
//     Pos_t new_x2 = 0.5f;

//     // Move the particles to be annihilated to the center of the cell
//     ptc.x1[idx] = new_x1;
//     ptc.x2[idx] = new_x2;

//     // Deposit extra current due to this movement
//     if (!check_bit(flag, ParticleFlag::ignore_current)) {
//       // Scalar djz[spline_t::support + 1][spline_t::support + 1] =
//       // {0.0f};
//       Scalar weight = -dev_charges[sp] * w;

//       Scalar djy[3] = {0.0f};
//       for (int j = -1; j <= 0; j++) {
//         Scalar sy0 = interp.interpolate(-old_x2 + j + 1);
//         Scalar sy1 = interp.interpolate(-new_x2 + j + 1);

//         size_t j_offset = (j + c2) * j1.pitch;
//         Scalar djx = 0.0f;
//         for (int i = -1; i <= 0; i++) {
//           Scalar sx0 = interp.interpolate(-old_x1 + i + 1);
//           Scalar sx1 = interp.interpolate(-new_x1 + i + 1);

//           // j1 is movement in r
//           int offset = j_offset + (i + c1) * sizeof(Scalar);
//           Scalar val0 = movement2d(sy0, sy1, sx0, sx1);
//           djx += val0;
//           atomicAdd(ptrAddr(j1, offset + sizeof(Scalar)), weight *
//           djx);

//           // j2 is movement in theta
//           Scalar val1 = movement2d(sx0, sx1, sy0, sy1);
//           djy[i + 1] += val1;
//           atomicAdd(ptrAddr(j2, offset + j2.pitch),
//                     weight * djy[i + 1]);
//         }
//       }
//     }

//     // Actually kill the particle
//     ptc.cell[idx] = MAX_CELL;
//   }
// }

// __global__ void
// flag_annihilation(particle_data ptc, size_t num, cudaPitchedPtr
// dens_e,
//                   cudaPitchedPtr dens_p, cudaPitchedPtr balance) {
//   for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num;
//        i += blockDim.x * gridDim.x) {
//     auto c = ptc.cell[i];
//     // Skip empty particles
//     if (c == MAX_CELL) continue;

//     auto flag = ptc.flag[i];
//     if (get_ptc_type(flag) > 1) continue;  // ignore ions
//     auto w = ptc.weight[i];

//     int c1 = dev_mesh.get_c1(c);
//     int c2 = dev_mesh.get_c2(c);
//     Scalar sin_t = std::sin(dev_mesh.pos(1, c2, 0.5f));
//     // size_t offset = c1 * sizeof(Scalar) + c2 * dens.pitch;

//     Scalar n_e = *ptrAddr(dens_e, c1, c2);
//     Scalar n_p = *ptrAddr(dens_p, c1, c2);
//     if (get_ptc_type(flag) == (int)ParticleType::electron)
//       n_e = atomicAdd(ptrAddr(dens_e, c1, c2), w);
//     else if (get_ptc_type(flag) == (int)ParticleType::positron)
//       n_p = atomicAdd(ptrAddr(dens_p, c1, c2), w);
//     Scalar r = std::exp(dev_mesh.pos(0, c1, 0.5f));
//     Scalar n_min = 0.1 * square(dev_mesh.inv_delta[0]) * sin_t;
//     // TODO: implement the proper condition
//     if (n_e > n_min && n_p > n_min) {
//       set_bit(flag, ParticleFlag::annihilate);
//       atomicAdd(ptrAddr(balance, c1, c2),
//                 w * (get_ptc_type(flag) ==
//                 (int)ParticleType::electron
//                          ? -1.0f
//                          : 1.0f));
//     }
//   }
//   // After this operation, the balance array will contain how much
//   // imbalance is there in the annihilated part. We will add this
//   // imbalance back in as an extra particle
// }

__global__ void
add_extra_particles(particle_data ptc, size_t num,
                    cudaPitchedPtr balance) {
  int t1 = blockIdx.x, t2 = blockIdx.y;
  int c1 = threadIdx.x, c2 = threadIdx.y;
  int n1 = dev_mesh.guard[0] + t1 * blockDim.x + c1;
  int n2 = dev_mesh.guard[1] + t2 * blockDim.y + c2;
  int num_offset = n2 * dev_mesh.dims[0] + n1;

  ptc.cell[num + num_offset] = num_offset;
  ptc.x1[num + num_offset] = 0.5f;
  ptc.x2[num + num_offset] = 0.5f;
  ptc.x3[num + num_offset] = 0.0f;
  ptc.p1[num + num_offset] = 0.0f;
  ptc.p2[num + num_offset] = 0.0f;
  ptc.p3[num + num_offset] = 0.0f;
  ptc.E[num + num_offset] = 1.0f;
  Scalar w = *ptrAddr(balance, n1, n2);
  ptc.weight[num + num_offset] = std::abs(w);
  if (w > 0)
    ptc.flag[num + num_offset] =
        set_ptc_type_flag(0, ParticleType::positron);
  else
    ptc.flag[num + num_offset] =
        set_ptc_type_flag(0, ParticleType::electron);
}

__global__ void
filter_current(cudaPitchedPtr j, cudaPitchedPtr j_tmp,
               cudaPitchedPtr A) {
  // Load position parameters
  int t1 = blockIdx.x, t2 = blockIdx.y;
  int c1 = threadIdx.x, c2 = threadIdx.y;
  int n1 = dev_mesh.guard[0] + t1 * blockDim.x + c1;
  int n2 = dev_mesh.guard[1] + t2 * blockDim.y + c2;
  size_t globalOffset = n2 * j.pitch + n1 * sizeof(Scalar);

  size_t dr_plus =
      (n1 < dev_mesh.dims[0] - dev_mesh.guard[0] - 1 ? sizeof(Scalar) : 0);
  size_t dr_minus =
      (n1 > dev_mesh.guard[0] ? sizeof(Scalar) : 0);
  size_t dt_plus =
      (n2 < dev_mesh.dims[1] - dev_mesh.guard[1] - 1 ? j.pitch : 0);
  size_t dt_minus =
      (n2 > dev_mesh.guard[1] ? j.pitch : 0);
  // Do the actual computation here
  (*ptrAddr(j_tmp, globalOffset)) =
      0.25f * *ptrAddr(j, globalOffset) * *ptrAddr(A, globalOffset);
  (*ptrAddr(j_tmp, globalOffset)) +=
      0.125f * *ptrAddr(j, globalOffset + dr_plus) *
      *ptrAddr(A, globalOffset + dr_plus);
  (*ptrAddr(j_tmp, globalOffset)) +=
      0.125f * *ptrAddr(j, globalOffset - dr_minus) *
      *ptrAddr(A, globalOffset - dr_minus);
  (*ptrAddr(j_tmp, globalOffset)) +=
      0.125f * *ptrAddr(j, globalOffset + dt_plus) *
      *ptrAddr(A, globalOffset + dt_plus);
  (*ptrAddr(j_tmp, globalOffset)) +=
      0.125f * *ptrAddr(j, globalOffset - dt_minus) *
      *ptrAddr(A, globalOffset - dt_minus);
  (*ptrAddr(j_tmp, globalOffset)) +=
      0.0625f * *ptrAddr(j, globalOffset + dr_plus + dt_plus) *
      *ptrAddr(A, globalOffset + dr_plus + dt_plus);
  (*ptrAddr(j_tmp, globalOffset)) +=
      0.0625f * *ptrAddr(j, globalOffset - dr_minus + dt_plus) *
      *ptrAddr(A, globalOffset - dr_minus + dt_plus);
  (*ptrAddr(j_tmp, globalOffset)) +=
      0.0625f * *ptrAddr(j, globalOffset + dr_plus - dt_minus) *
      *ptrAddr(A, globalOffset + dr_plus - dt_minus);
  (*ptrAddr(j_tmp, globalOffset)) +=
      0.0625f * *ptrAddr(j, globalOffset - dr_minus - dt_minus) *
      *ptrAddr(A, globalOffset - dr_minus - dt_minus);
  (*ptrAddr(j_tmp, globalOffset)) /= *ptrAddr(A, globalOffset);
}

}  // namespace Kernels

PtcUpdaterLogSph::PtcUpdaterLogSph(const cu_sim_environment &env)
    : PtcUpdaterDev(env),
      d_rand_states(nullptr),
      m_threadsPerBlock(256),
      m_blocksPerGrid(256),
      m_dens_e(env.local_grid()),
      m_dens_p(env.local_grid()),
      m_balance(env.local_grid()),
      m_annihilate(env.local_grid().extent()) {
  const Grid_LogSph_dev &grid =
      dynamic_cast<const Grid_LogSph_dev &>(env.grid());
  m_mesh_ptrs = grid.get_mesh_ptrs();

  int seed = m_env.params().random_seed;
  CudaSafeCall(cudaMalloc(
      &d_rand_states,
      m_threadsPerBlock * m_blocksPerGrid * sizeof(curandState)));
  init_rand_states((curandState *)d_rand_states, seed,
                   m_threadsPerBlock, m_blocksPerGrid);

  // m_J1.initialize();
  // m_J2.initialize();
}

PtcUpdaterLogSph::~PtcUpdaterLogSph() {
  cudaFree((curandState *)d_rand_states);
}

void
PtcUpdaterLogSph::update_particles(cu_sim_data &data, double dt,
                                   uint32_t step) {
  initialize_dev_fields(data);

  if (m_env.grid().dim() == 2) {
    auto &mesh = m_env.grid().mesh();
    data.photon_flux.assign_dev(0.0);
    // Skip empty particle array
    if (data.particles.number() > 0) {
      Logger::print_info(
          "Updating {} particles in log spherical coordinates",
          data.particles.number());
      Kernels::vay_push_2d<<<256, 256>>>(
          data.particles.data(), data.particles.number(), m_dev_fields,
          data.photon_flux.data_d(), (curandState *)d_rand_states, dt);
      CudaCheckError();
      data.J.initialize();
      for (auto &rho : data.Rho) {
        rho.initialize();
      }
      // m_J1.initialize();
      // m_J2.initialize();
      Kernels::deposit_current_2d_log_sph<<<256, 512>>>(
          data.particles.data(), data.particles.number(), m_dev_fields,
          m_mesh_ptrs, dt);
      CudaCheckError();
      Kernels::process_j<<<dim3(32, 32), dim3(32, 32)>>>(
          m_dev_fields, m_mesh_ptrs, dt);
      CudaCheckError();

      // Kernels::convert_j<<<dim3(32, 32), dim3(32, 32)>>>(
      //     m_J1.ptr(), m_J2.ptr(), m_dev_fields);
      // CudaCheckError();
      dim3 blockSize(32, 16);
      dim3 gridSize(mesh.reduced_dim(0) / 32, mesh.reduced_dim(1) / 16);
      for (int n = 0; n < m_env.params().current_smoothing; n++) {
        Kernels::filter_current<<<gridSize, blockSize>>>(
            data.J.ptr(0), m_dens_e.ptr(), m_mesh_ptrs.A1_e);
        data.J.data(0).copy_from(m_dens_e.data());
        Kernels::filter_current<<<gridSize, blockSize>>>(
            data.J.ptr(1), m_dens_e.ptr(), m_mesh_ptrs.A2_e);
        data.J.data(1).copy_from(m_dens_e.data());
        for (int sp = 0; sp < m_env.params().num_species; sp++) {
          Kernels::filter_current<<<gridSize, blockSize>>>(
              data.Rho[sp].ptr(), m_dens_e.ptr(), m_mesh_ptrs.dV);
          data.Rho[sp].data().copy_from(m_dens_e.data());
        }
        CudaCheckError();
      }
    }
    // Skip empty particle array
    if (data.photons.number() > 0) {
      Logger::print_info(
          "Updating {} photons in log spherical coordinates",
          data.photons.number());
      Kernels::move_photons<<<256, 512>>>(data.photons.data(),
                                          data.photons.number(), dt);
      CudaCheckError();
    }
  }
  cudaDeviceSynchronize();
}

void
PtcUpdaterLogSph::handle_boundary(cu_sim_data &data) {
  data.particles.clear_guard_cells();
  data.photons.clear_guard_cells();

  Kernels::boundary_rho<<<32, 512>>>(m_dev_fields, m_mesh_ptrs);
  CudaCheckError();
  cudaDeviceSynchronize();
}

void
PtcUpdaterLogSph::inject_ptc(cu_sim_data &data, int inj_per_cell,
                             Scalar p1, Scalar p2, Scalar p3, Scalar w,
                             Scalar omega) {
  Kernels::inject_ptc<<<m_blocksPerGrid, m_threadsPerBlock>>>(
      data.particles.data(), data.particles.number(), inj_per_cell, p1,
      p2, p3, w, (curandState *)d_rand_states, omega);
  CudaCheckError();

  data.particles.set_num(data.particles.number() +
                         2 * inj_per_cell *
                             data.E.grid().mesh().reduced_dim(1));
}

void
PtcUpdaterLogSph::annihilate_extra_pairs(cu_sim_data &data, double dt) {
  m_dens_e.data().assign_dev(0.0);
  m_dens_p.data().assign_dev(0.0);
  m_annihilate.assign_dev(0);

  Kernels::check_annihilation<<<256, 512>>>(
      data.particles.data(), data.particles.number(), m_dens_e.ptr(),
      m_dens_p.ptr(), m_annihilate.data_d());
  CudaCheckError();

  m_dens_e.data().assign_dev(0.0);
  m_dens_p.data().assign_dev(0.0);
  m_balance.data().assign_dev(0.0);
  Kernels::flag_annihilation<<<256, 512>>>(
      data.particles.data(), data.particles.number(), m_dens_e.ptr(),
      m_dens_p.ptr(), m_balance.ptr(), m_annihilate.data_d());
  CudaCheckError();

  Kernels::annihilate_pairs<<<256, 512>>>(
      data.particles.data(), data.particles.number(), data.J.ptr(0),
      data.J.ptr(1), data.J.ptr(2), m_mesh_ptrs, dt);
  CudaCheckError();

  auto &mesh = data.E.grid().mesh();
  dim3 blockSize(32, 16);
  dim3 gridSize(mesh.reduced_dim(0) / 32, mesh.reduced_dim(1) / 16);

  Kernels::add_extra_particles<<<gridSize, blockSize>>>(
      data.particles.data(), data.particles.number(), m_balance.ptr());
  CudaCheckError();

  cudaDeviceSynchronize();
  data.particles.set_num(data.particles.number() +
                         mesh.reduced_dim(0) * mesh.reduced_dim(1));
}

}  // namespace Aperture