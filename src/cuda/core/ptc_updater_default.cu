#include "core/detail/multi_array_utils.hpp"
#include "cuda/constant_mem.h"
#include "cuda/core/cu_sim_data.h"
#include "cuda/core/ptc_updater_dev.h"
#include "cuda/core/ptc_updater_helper.cuh"
#include "cuda/core/cu_sim_environment.h"
#include "cuda/cudaUtility.h"
#include "cuda/kernels.h"
#include "cuda/ptr_util.h"
#include "cuda/utils/interpolation.cuh"
#include "cuda/utils/iterate_devices.h"
#include "utils/logger.h"
#include "utils/util_functions.h"

#define DEPOSIT_EPS 1.0e-10f

namespace Aperture {

namespace Kernels {

HOST_DEVICE void
vay_push_2d(Scalar &p1, Scalar &p2, Scalar &p3, Scalar &gamma,
            PtcUpdaterDev::fields_data &fields, Pos_t x1, Pos_t x2,
            int c1, int c2, Scalar q_over_m) {
  Interpolator2D<spline_t> interp;
  Scalar E1 =
      interp(fields.E1, x1, x2, c1, c2, Stagger(0b001)) * q_over_m;
  Scalar E2 =
      interp(fields.E2, x1, x2, c1, c2, Stagger(0b010)) * q_over_m;
  Scalar E3 =
      interp(fields.E3, x1, x2, c1, c2, Stagger(0b100)) * q_over_m;
  Scalar B1 =
      interp(fields.B1, x1, x2, c1, c2, Stagger(0b110)) * q_over_m;
  Scalar B2 =
      interp(fields.B2, x1, x2, c1, c2, Stagger(0b101)) * q_over_m;
  Scalar B3 =
      interp(fields.B3, x1, x2, c1, c2, Stagger(0b011)) * q_over_m;

  // step 1: Update particle momentum using vay pusher
  Scalar up1 = p1 + 2.0f * E1 + (p2 * B3 - p3 * B2) / gamma;
  Scalar up2 = p2 + 2.0f * E2 + (p3 * B1 - p1 * B3) / gamma;
  Scalar up3 = p3 + 2.0f * E3 + (p1 * B2 - p2 * B1) / gamma;
  Scalar tt = B1 * B1 + B2 * B2 + B3 * B3;
  Scalar ut = up1 * B1 + up2 * B3 + up3 * B3;

  Scalar sigma = 1.0f + up1 * up1 + up2 * up2 + up3 * up3 - tt;
  Scalar inv_gamma2 =
      2.0f / (sigma + std::sqrt(sigma * sigma + 4.0f * (tt + ut * ut)));
  Scalar s = 1.0f / (1.0f + inv_gamma2 * tt);
  gamma = 1.0f / std::sqrt(inv_gamma2);

  p1 = (up1 + B1 * ut * inv_gamma2 + (up2 * B3 - up3 * B2) / gamma) * s;
  p2 = (up2 + B2 * ut * inv_gamma2 + (up3 * B1 - up1 * B3) / gamma) * s;
  p3 = (up3 + B3 * ut * inv_gamma2 + (up1 * B2 - up2 * B1) / gamma) * s;
}

HOST_DEVICE void
ptc_movement_2d(Pos_t &new_x1, Pos_t &new_x2, Pos_t &new_x3, Scalar p1,
                Scalar p2, Scalar p3, Scalar gamma, Scalar dt) {
  new_x1 += dt * p1 / gamma;
  new_x2 += dt * p2 / gamma;
  new_x3 += dt * p3 / gamma;
}

#define MIN_BLOCKS_PER_MP 3
#define MAX_THREADS_PER_BLOCK 256
__global__ void
// __launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP)
__launch_bounds__(128, 4)
    update_particles(particle_data ptc, size_t num,
                     PtcUpdaterDev::fields_data fields, Scalar dt) {
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num;
       idx += blockDim.x * gridDim.x) {
    auto c = ptc.cell[idx];
    // Skip empty particles
    if (c == MAX_CELL) continue;

    // Load particle quantities
    Interpolator3D<spline_t> interp;
    auto flag = ptc.flag[idx];
    int sp = get_ptc_type(flag);
    // auto w = ptc.weight[idx];
    auto old_x1 = ptc.x1[idx], old_x2 = ptc.x2[idx],
         old_x3 = ptc.x3[idx];
    auto p1 = ptc.p1[idx], p2 = ptc.p2[idx], p3 = ptc.p3[idx];
    int c1 = dev_mesh.get_c1(c);
    int c2 = dev_mesh.get_c2(c);
    int c3 = dev_mesh.get_c3(c);
    Scalar q_over_m = dt * 0.5f * dev_charges[sp] / dev_masses[sp];

    // step 0: Grab E & M fields at the particle position
    Scalar gamma = std::sqrt(1.0f + p1 * p1 + p2 * p2 + p3 * p3);
    if (!check_bit(flag, ParticleFlag::ignore_EM)) {
      Scalar E1 = interp(fields.E1, old_x1, old_x2, old_x3, c1, c2, c3,
                         Stagger(0b001)) *
                  q_over_m;
      Scalar E2 = interp(fields.E2, old_x1, old_x2, old_x3, c1, c2, c3,
                         Stagger(0b010)) *
                  q_over_m;
      Scalar E3 = interp(fields.E3, old_x1, old_x2, old_x3, c1, c2, c3,
                         Stagger(0b100)) *
                  q_over_m;
      Scalar B1 = interp(fields.B1, old_x1, old_x2, old_x3, c1, c2, c3,
                         Stagger(0b110)) *
                  q_over_m;
      Scalar B2 = interp(fields.B2, old_x1, old_x2, old_x3, c1, c2, c3,
                         Stagger(0b101)) *
                  q_over_m;
      Scalar B3 = interp(fields.B3, old_x1, old_x2, old_x3, c1, c2, c3,
                         Stagger(0b011)) *
                  q_over_m;

      // step 1: Update particle momentum using vay pusher
      Scalar up1 = p1 + 2.0f * E1 + (p2 * B3 - p3 * B2) / gamma;
      Scalar up2 = p2 + 2.0f * E2 + (p3 * B1 - p1 * B3) / gamma;
      Scalar up3 = p3 + 2.0f * E3 + (p1 * B2 - p2 * B1) / gamma;
      Scalar tt = B1 * B1 + B2 * B2 + B3 * B3;
      Scalar ut = up1 * B1 + up2 * B3 + up3 * B3;

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
      ptc.p1[idx] = p1;
      ptc.p2[idx] = p2;
      ptc.p3[idx] = p3;
    }
  }
}

__global__ void
__launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP)
    deposit_current_3d(particle_data ptc, size_t num,
                       PtcUpdaterDev::fields_data fields, Scalar dt) {
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num;
       idx += blockDim.x * gridDim.x) {
    auto c = ptc.cell[idx];
    // Skip empty particles
    if (c == MAX_CELL) continue;

    // Load particle quantities
    Interpolator3D<spline_t> interp;
    int c1 = dev_mesh.get_c1(c);
    int c2 = dev_mesh.get_c2(c);
    int c3 = dev_mesh.get_c3(c);
    auto p1 = ptc.p1[idx], p2 = ptc.p2[idx], p3 = ptc.p3[idx];
    auto flag = ptc.flag[idx];
    int sp = get_ptc_type(flag);
    auto w = ptc.weight[idx];
    auto old_x1 = ptc.x1[idx], old_x2 = ptc.x2[idx],
         old_x3 = ptc.x3[idx];

    // step 0: Grab E & M fields at the particle position
    Scalar gamma = std::sqrt(1.0f + p1 * p1 + p2 * p2 + p3 * p3);

    // step 2: Compute particle movement and update position
    Pos_t new_x1 = old_x1 + dt * p1 / gamma;
    Pos_t new_x2 = old_x2 + dt * p2 / gamma;
    Pos_t new_x3 = old_x3 + dt * p3 / gamma;
    int dc1 = floor(new_x1);
    int dc2 = floor(new_x2);
    int dc3 = floor(new_x3);
    ptc.cell[idx] = dev_mesh.get_idx(c1 + dc1, c2 + dc2, c3 + dc3);
    new_x1 -= (Pos_t)dc1;
    new_x2 -= (Pos_t)dc2;
    new_x3 -= (Pos_t)dc3;
    ptc.x1[idx] = new_x1;
    ptc.x2[idx] = new_x2;
    ptc.x3[idx] = new_x3;

    // step 3: Deposit current
    if (check_bit(flag, ParticleFlag::ignore_current)) continue;
    // Scalar djz[spline_t::support + 1][spline_t::support + 1] =
    // {0.0f}; Scalar djy[spline_t::support + 1][spline_t::support + 1]
    // = {0.0f};
    Scalar wdt = -dev_charges[sp] * w / dt;
    int sup2 = interp.support() + 2;
    int sup23 = sup2 * sup2 * sup2;
    for (int k = 0; k < sup2; k++) {
      int kk = (((idx + k) % sup23) / (sup2 * sup2)) - interp.radius();
      Scalar sz0 = interp.interpolate(0.5f - old_x3 + kk);
      Scalar sz1 = interp.interpolate(0.5f - new_x3 + (kk + dc3));
      if (std::abs(sz0) < DEPOSIT_EPS && std::abs(sz1) < DEPOSIT_EPS)
        continue;
      size_t k_offset = (kk + c3) * fields.J1.pitch * fields.J1.ysize;
      for (int j = 0; j < sup2; j++) {
        // int jj = j;
        int jj = (((idx + j) % sup23) / sup2) % sup2 - interp.radius();
        Scalar sy0 = interp.interpolate(0.5f - old_x2 + jj);
        Scalar sy1 = interp.interpolate(0.5f - new_x2 + (jj + dc2));
        // if (std::abs(sy0) < DEPOSIT_EPS && std::abs(sy1) <
        // DEPOSIT_EPS)
        //   continue;
        size_t j_offset = (jj + c2) * fields.J1.pitch;
        // Scalar djx = 0.0f;
        for (int i = 0; i < sup2; i++) {
          // int ii = i;
          int ii = ((idx + i) % sup23) % sup2 - interp.radius();
          Scalar sx0 = interp.interpolate(0.5f - old_x1 + ii);
          Scalar sx1 = interp.interpolate(0.5f - new_x1 + (ii + dc1));
          // if (std::abs(sx0) < DEPOSIT_EPS && std::abs(sx1) <
          // DEPOSIT_EPS)
          //   continue;

          int offset = k_offset + j_offset + (ii + c1) * sizeof(Scalar);
          // djx -= wdt * dev_mesh.delta[0] *
          //        movement3d(sy0, sy1, sz0, sz1, sx0, sx1);
          // djy[k + interp.radius() + 1][i + interp.radius() + 1] -=
          //     wdt * dev_mesh.delta[1] *
          //     movement3d(sz0, sz1, sx0, sx1, sy0, sy1);
          // djz[i + interp.radius() + 1][j + interp.radius() + 1] -=
          //     wdt * dev_mesh.delta[2] *
          //     movement3d(sx0, sx1, sy0, sy1, sz0, sz1);
          Scalar val0 = movement3d(sy0, sy1, sz0, sz1, sx0, sx1);
          if (std::abs(val0) > 0.0f)
            atomicAdd((Scalar *)((char *)fields.J1.ptr + offset),
                      wdt * dev_mesh.delta[0] * val0);
          Scalar val1 = movement3d(sz0, sz1, sx0, sx1, sy0, sy1);
          if (std::abs(val1) > 0.0f)
            atomicAdd((Scalar *)((char *)fields.J2.ptr + offset),
                      wdt * dev_mesh.delta[1] * val1);
          Scalar val2 = movement3d(sx0, sx1, sy0, sy1, sz0, sz1);
          if (std::abs(val2) > 0.0f)
            atomicAdd((Scalar *)((char *)fields.J3.ptr + offset),
                      wdt * dev_mesh.delta[2] * val2);
          Scalar s1 = sx1 * sy1 * sz1;
          if (std::abs(s1) > 0.0f)
            atomicAdd((Scalar *)((char *)fields.Rho[sp].ptr + offset),
                      s1 * dev_charges[sp]);
        }
      }
    }
  }
}

__global__ void
update_particles_2d(particle_data ptc, size_t num,
                    PtcUpdaterDev::fields_data fields, Scalar dt) {
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num;
       idx += blockDim.x * gridDim.x) {
    auto c = ptc.cell[idx];
    // Skip empty particles
    if (c == MAX_CELL) continue;

    // Load particle quantities
    Interpolator2D<spline_t> interp;
    auto flag = ptc.flag[idx];
    int sp = get_ptc_type(flag);
    auto w = ptc.weight[idx];
    auto old_x1 = ptc.x1[idx], old_x2 = ptc.x2[idx],
         old_x3 = ptc.x3[idx];
    auto p1 = ptc.p1[idx], p2 = ptc.p2[idx], p3 = ptc.p3[idx];
    int c1 = dev_mesh.get_c1(c);
    int c2 = dev_mesh.get_c2(c);
    Scalar q_over_m = dt * 0.5f * dev_charges[sp] / dev_masses[sp];
    // step 0: Grab E & M fields at the particle position
    Scalar gamma = std::sqrt(1.0f + p1 * p1 + p2 * p2 + p3 * p3);
    if (!check_bit(flag, ParticleFlag::ignore_EM)) {
      vay_push_2d(p1, p2, p3, gamma, fields, old_x1, old_x2, c1, c2,
                  q_over_m);
      ptc.p1[idx] = p1;
      ptc.p2[idx] = p2;
      ptc.p3[idx] = p3;
    }

    // step 2: Compute particle movement and update position
    Pos_t new_x1 = old_x1, new_x2 = old_x2, new_x3 = old_x3;
    ptc_movement_2d(new_x1, new_x2, new_x3, p1, p2, p3, gamma, dt);
    int dc1 = floor(new_x1);
    int dc2 = floor(new_x2);
    new_x1 -= (Pos_t)dc1;
    new_x2 -= (Pos_t)dc2;
    ptc.cell[idx] = dev_mesh.get_idx(c1 + dc1, c2 + dc2);
    ptc.x1[idx] = new_x1;
    ptc.x2[idx] = new_x2;
    ptc.x3[idx] = old_x3 + dt * p3 / gamma;

    // step 3: Deposit current
    if (!check_bit(flag, ParticleFlag::ignore_current)) {
      Scalar wdt = -dev_charges[sp] * w / dt;
      int sup2 = interp.support() + 2;
      int sup22 = sup2 * sup2;
      for (int j = 0; j < sup2; j++) {
        // int jj = j;
        int jj = (((idx + j) % sup22) / sup2);
        Scalar sy0 =
            interp.interpolate(0.5f + jj - interp.radius() - old_x2);
        Scalar sy1 = interp.interpolate(0.5f + jj - interp.radius() +
                                        dc2 - new_x2);
        size_t j_offset = (jj - interp.radius() + c2) * fields.J1.pitch;
        // Scalar djx = 0.0f;
        for (int i = 0; i < sup2; i++) {
          // int ii = i;
          int ii = ((idx + i) % sup22) % sup2;
          Scalar sx0 =
              interp.interpolate(0.5f + ii - interp.radius() - old_x1);
          Scalar sx1 = interp.interpolate(0.5f + ii - interp.radius() +
                                          dc1 - new_x1);

          int offset = j_offset + (ii + c1) * sizeof(Scalar);
          // djx -= wdt * dev_mesh.delta[0] *
          //        movement3d(sy0, sy1, sz0, sz1, sx0, sx1);
          // djy[k + interp.radius() + 1][i + interp.radius() + 1] -=
          //     wdt * dev_mesh.delta[1] *
          //     movement3d(sz0, sz1, sx0, sx1, sy0, sy1);
          // djz[i + interp.radius() + 1][j + interp.radius() + 1] -=
          //     wdt * dev_mesh.delta[2] *
          //     movement3d(sx0, sx1, sy0, sy1, sz0, sz1);
          atomicAdd(
              (Scalar *)((char *)fields.J1.ptr + offset),
              wdt * dev_mesh.delta[0] * movement2d(sy0, sy1, sx0, sx1));
          atomicAdd(
              (Scalar *)((char *)fields.J2.ptr + offset),
              wdt * dev_mesh.delta[1] * movement2d(sx0, sx1, sy0, sy1));
          atomicAdd((Scalar *)((char *)fields.J3.ptr + offset),
                    dev_charges[sp] * w * (p3 / gamma) *
                        center2d(sx0, sx1, sy0, sy1));
          atomicAdd((Scalar *)((char *)fields.Rho[sp].ptr + offset),
                    sx1 * sy1 * dev_charges[sp]);
        }
      }
    }
  }
}

__global__ void
update_particles_1d(particle_data ptc, size_t num, const Scalar *E1,
                    Scalar *J1, Scalar *Rho, Scalar dt) {
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num;
       idx += blockDim.x * gridDim.x) {
    auto cell = ptc.cell[idx];
    // Skip empty particles
    if (cell == MAX_CELL) continue;

    auto old_x1 = ptc.x1[idx];
    auto p1 = ptc.p1[idx];
    int sp = get_ptc_type(ptc.flag[idx]);
    auto flag = ptc.flag[idx];
    auto w = ptc.weight[idx];

    // step 1: Update particle momentum
    if (!check_bit(flag, ParticleFlag::ignore_EM)) {
      Scalar E = E1[cell] * old_x1 + E1[cell - 1] * (1.0f - old_x1);
      p1 += dev_charges[sp] * E * dt / dev_masses[sp];
      ptc.p1[idx] = p1;
    }

    // step 2: Compute particle movement and update position
    Scalar gamma = std::sqrt(1.0f + p1 * p1);
    Scalar dx1 = dt * p1 / (gamma * dev_mesh.delta[0]);
    Scalar new_x1 = old_x1 + dx1;
    int delta_cell = floor(new_x1);
    cell += delta_cell;

    ptc.cell[idx] = cell;
    ptc.x1[idx] = new_x1 - (Pos_t)delta_cell;

    // step 3: Deposit current
    Scalar s0, s1, j_sum = 0.0f;
    for (int c = cell - delta_cell - 2; c <= cell - delta_cell + 2;
         c++) {
      s1 = interpolate(new_x1, cell, c);

      if (!check_bit(flag, ParticleFlag::ignore_current)) {
        s0 = interpolate(old_x1, cell - delta_cell, c);
        j_sum += dev_charges[sp] * w * (s0 - s1) * dev_mesh.delta[0] /
                 dev_params.delta_t;
        atomicAdd(&J1[c], j_sum);
      }
      atomicAdd(&Rho[c], dev_charges[sp] * w * s1);
    }
  }
}

}  // namespace Kernels

}  // namespace Aperture