#include "cuda/constant_mem.h"
#include "cuda/cudaUtility.h"
#include "cuda/kernels.h"
#include "cuda/ptr_util.h"
#include "data/detail/multi_array_utils.hpp"
#include "ptc_updater_helper.cuh"
#include "ptc_updater_logsph.h"
#include "cu_sim_data.h"
#include "sim_environment_dev.h"
#include "utils/interpolation.cuh"
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
vay_push_2d(particle_data ptc, size_t num, fields_data fields,
            Scalar dt) {
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

      // printf("p after is (%f, %f, %f), gamma is %f, inv_gamma2 is %f, %d\n", p1, p2, p3,
      //        gamma, inv_gamma2, dev_params.gravity_on);
      // Add an artificial gravity
      if (dev_params.gravity_on) {
        Scalar r = exp(dev_mesh.pos(0, c1, old_x1));
        p1 -= dt * dev_params.gravity / (r * r);
        gamma = sqrt(1.0f + p1 * p1 + p2 * p2 + p3 * p3);
      }

      // printf("gamma after is %f\n", gamma);
      // printf("p before is (%f, %f, %f)\n", ptc.p1[idx], ptc.p2[idx],
      // ptc.p3[idx]);
      ptc.p1[idx] = p1;
      ptc.p2[idx] = p2;
      ptc.p3[idx] = p3;
      ptc.E[idx] = gamma;
    }
  }
}

__global__ void
boris_push_2d(particle_data ptc, size_t num, fields_data fields,
              Scalar dt) {
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num;
       idx += blockDim.x * gridDim.x) {
    auto c = ptc.cell[idx];
    // Skip empty particles
    if (c == MAX_CELL) continue;

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
    if (!check_bit(flag, ParticleFlag::ignore_EM)) {
      Scalar E1 =
          (interp(fields.E1, old_x1, old_x2, c1, c2, Stagger(0b110)) +
           interp(dev_bg_fields.E1, old_x1, old_x2, c1, c2,
                  Stagger(0b110))) *
          q_over_m;
      Scalar E2 =
          (interp(fields.E2, old_x1, old_x2, c1, c2, Stagger(0b101)) +
           interp(dev_bg_fields.E2, old_x1, old_x2, c1, c2,
                  Stagger(0b101))) *
          q_over_m;
      Scalar E3 =
          (interp(fields.E3, old_x1, old_x2, c1, c2, Stagger(0b011)) +
           interp(dev_bg_fields.E3, old_x1, old_x2, c1, c2,
                  Stagger(0b011))) *
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
          (interp(fields.B3, old_x1, old_x2, c1, c2, Stagger(0b100)) +
           interp(dev_bg_fields.B3, old_x1, old_x2, c1, c2,
                  Stagger(0b100))) *
          q_over_m;

      // printf("B is (%f, %f, %f)\n", B1, B2, B3);
      // printf("p is (%f, %f, %f)\n", p1, p2, p3);
      // printf("B cell is %f\n", *ptrAddr(fields.B1, c1*sizeof(Scalar)
      // + c2*fields.B1.pitch)); printf("q over m is %f\n", q_over_m);

      // step 1: Update particle momentum using boris pusher
      Scalar pm1 = p1 + E1;
      Scalar pm2 = p2 + E2;
      Scalar pm3 = p3 + E3;
      // printf("pm is (%f, %f, %f)\n", pm1, pm2, pm3);
      Scalar gamma =
          std::sqrt(1.0f + pm1 * pm1 + pm2 * pm2 + pm3 * pm3);
      Scalar pp1 = pm1 + (pm2 * B3 - pm3 * B2) / gamma;
      Scalar pp2 = pm2 + (pm3 * B1 - pm1 * B3) / gamma;
      Scalar pp3 = pm3 + (pm1 * B2 - pm2 * B1) / gamma;
      // printf("pp is (%f, %f, %f)\n", pp1, pp2, pp3);
      Scalar t2p1 =
          1.0f + (B1 * B1 + B2 * B2 + B3 * B3) / (gamma * gamma);
      // printf("t2p1 is %f, gamma is %f\n", t2p1, gamma);
      // printf("cross3 are: %f, %f\n", (pm1 * B2 - pm2 * B1) / gamma,
      //        2.0f * (pp1 * B2 - pp2 * B1) / t2p1);

      // p1 -= 10.0*dt;
      p1 = E1 + pm1 + 2.0f * (pp2 * B3 - pp3 * B2) / t2p1;
      p2 = E2 + pm2 + 2.0f * (pp3 * B1 - pp1 * B3) / t2p1;
      p3 = E3 + pm3 + 2.0f * (pp1 * B2 - pp2 * B1) / t2p1;
      // printf("p is (%f, %f, %f)\n", p1, p2, p3);
      ptc.p1[idx] = p1;
      ptc.p2[idx] = p2;
      ptc.p3[idx] = p3;
      ptc.E[idx] = gamma;
    }
  }
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
                               fields_data fields,
                               Grid_LogSph::mesh_ptrs mesh_ptrs,
                               cudaPitchedPtr j1, cudaPitchedPtr j2,
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
                  -weight * v3 * val2 /
                      *ptrAddr(mesh_ptrs.dV, offset));

        // rho is deposited at the final position
        Scalar s1 = sx1 * sy1;
        atomicAdd(ptrAddr(fields.Rho[sp], offset), -weight * s1);
      }
    }
  }
}

__global__ void
convert_j(cudaPitchedPtr j1, cudaPitchedPtr j2, fields_data fields) {
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
process_j(fields_data fields, Grid_LogSph::mesh_ptrs mesh_ptrs,
          Scalar dt) {
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
  for (int
           i = dev_mesh.guard[1] + 1 + id;
           // i = dev_mesh.dims[1] - dev_mesh.guard[1] - 3 + id;
       i < dev_mesh.dims[1] - dev_mesh.guard[1] - 1;
       i += blockDim.x * gridDim.x) {
    size_t offset = num + i * inj_per_cell * 2;
    Scalar r = exp(dev_mesh.pos(0, dev_mesh.guard[0] + 2, 0.5f));
    for (int n = 0; n < inj_per_cell; n++) {
      Pos_t x2 = curand_uniform(&localState);
      Scalar theta = dev_mesh.pos(1, i, x2);
      Scalar vphi = omega * r * sin(theta);
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
      ptc.weight[offset + n * 2] = w * sin(theta);
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
      ptc.weight[offset + n * 2 + 1] = w * sin(theta);
      ptc.flag[offset + n * 2 + 1] = set_ptc_type_flag(
          bit_or(ParticleFlag::primary), ParticleType::positron);
    }
  }
  states[id] = localState;
}

__global__ void
boundary_rho(fields_data fields, Grid_LogSph::mesh_ptrs mesh_ptrs) {
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

}  // namespace Kernels

PtcUpdaterLogSph::PtcUpdaterLogSph(const Environment &env)
    : PtcUpdaterDev(env),
      d_rand_states(nullptr),
      m_threadsPerBlock(256),
      m_blocksPerGrid(128),
      m_J1(env.local_grid()),
      m_J2(env.local_grid()) {
  const Grid_LogSph &grid =
      dynamic_cast<const Grid_LogSph &>(env.grid());
  m_mesh_ptrs = grid.get_mesh_ptrs();

  int seed = m_env.params().random_seed;
  CudaSafeCall(cudaMalloc(
      &d_rand_states,
      m_threadsPerBlock * m_blocksPerGrid * sizeof(curandState)));
  init_rand_states((curandState *)d_rand_states, seed,
                   m_threadsPerBlock, m_blocksPerGrid);

  m_J1.initialize();
  m_J2.initialize();
}

PtcUpdaterLogSph::~PtcUpdaterLogSph() {
  cudaFree((curandState *)d_rand_states);
}

void
PtcUpdaterLogSph::update_particles(cu_sim_data &data, double dt) {
  initialize_dev_fields(data);

  if (m_env.grid().dim() == 2) {
    // Skip empty particle array
    if (data.particles.number() > 0) {
      Logger::print_info(
          "Updating {} particles in log spherical coordinates",
          data.particles.number());
      Kernels::vay_push_2d<<<256, 512>>>(data.particles.data(),
                                         data.particles.number(),
                                         m_dev_fields, dt);
      CudaCheckError();
      data.J.initialize();
      for (auto &rho : data.Rho) {
        rho.initialize();
      }
      // m_J1.initialize();
      // m_J2.initialize();
      Kernels::deposit_current_2d_log_sph<<<256, 512>>>(
          data.particles.data(), data.particles.number(),
          m_dev_fields, m_mesh_ptrs, m_J1.ptr(), m_J2.ptr(), dt);
      CudaCheckError();
      Kernels::process_j<<<dim3(32, 32), dim3(32, 32)>>>(
          m_dev_fields, m_mesh_ptrs, dt);
      CudaCheckError();

      // Kernels::convert_j<<<dim3(32, 32), dim3(32, 32)>>>(
      //     m_J1.ptr(), m_J2.ptr(), m_dev_fields);
      // CudaCheckError();
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
PtcUpdaterLogSph::inject_ptc(cu_sim_data &data, int inj_per_cell, Scalar p1,
                             Scalar p2, Scalar p3, Scalar w,
                             Scalar omega) {
  Kernels::inject_ptc<<<m_blocksPerGrid, m_threadsPerBlock>>>(
      data.particles.data(), data.particles.number(), inj_per_cell, p1,
      p2, p3, w, (curandState *)d_rand_states, omega);
  CudaCheckError();

  data.particles.set_num(data.particles.number() +
                         2 * inj_per_cell *
                             data.E.grid().mesh().reduced_dim(1));
}

}  // namespace Aperture