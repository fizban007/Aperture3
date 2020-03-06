#include "algorithms/ptc_updater.h"
#include "core/constant_defs.h"
#include "core/typedefs.h"
#include "cuda/algorithms/ptc_updater_helper.cuh"
#include "cuda/constant_mem.h"
#include "cuda/cudaUtility.h"
#include "cuda/data_ptrs.h"
#include "sim_data.h"
#include "sim_environment.h"
#include "utils/logger.h"
#include "utils/timer.h"
#include "utils/util_functions.h"

#include "sync_cooling.cuh"
#include "vay_push.cuh"

namespace Aperture {

namespace Kernels {

__global__ void
ptc_push_cart_1d(data_ptrs data, size_t num, Scalar dt) {
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num;
       idx += blockDim.x * gridDim.x) {
    auto &ptc = data.particles;

    auto c = ptc.cell[idx];
    // Skip empty particles
    if (c == MAX_CELL) continue;
    if (!dev_mesh.is_in_bulk(c)) {
      ptc.cell[idx] = MAX_CELL;
      continue;
    }
    // Load particle quantities
    Interpolator1D<spline_t> interp;
    auto flag = ptc.flag[idx];
    int sp = get_ptc_type(flag);
    auto x1 = ptc.x1[idx];
    auto p1 = ptc.p1[idx], p2 = ptc.p2[idx], p3 = ptc.p3[idx],
         gamma = ptc.E[idx];

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
      Scalar E1 = interp(data.E1, x1, c, Stagger(0b110)) * q_over_m;
      Scalar E2 = interp(data.E2, x1, c, Stagger(0b101)) * q_over_m;
      Scalar E3 = interp(data.E3, x1, c, Stagger(0b011)) * q_over_m;
      Scalar B1 = interp(data.B1, x1, c, Stagger(0b001)) * q_over_m;
      Scalar B2 = interp(data.B2, x1, c, Stagger(0b010)) * q_over_m;
      Scalar B3 = interp(data.B3, x1, c, Stagger(0b100)) * q_over_m;

      vay_push(p1, p2, p3, gamma, E1, E2, E3, B1, B2, B3, q_over_m, dt);

      // if (dev_params.rad_cooling_on && sp != (int)ParticleType::ion)
      // {
      //   sync_kill_perp(p1, p2, p3, gamma, B1, B2, B3, E1, E2, E3,
      //                  q_over_m);
      // }

      ptc.p1[idx] = p1;
      ptc.p2[idx] = p2;
      ptc.p3[idx] = p3;
      ptc.E[idx] = gamma;
    }
  }
}

__global__ void
ptc_push_cart_2d(data_ptrs data, size_t num, Scalar dt) {
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num;
       idx += blockDim.x * gridDim.x) {
    auto &ptc = data.particles;

    auto c = ptc.cell[idx];
    // Skip empty particles
    if (c == MAX_CELL) continue;
    if (!dev_mesh.is_in_bulk(c)) {
      ptc.cell[idx] = MAX_CELL;
      continue;
    }
    int c1 = dev_mesh.get_c1(c);
    int c2 = dev_mesh.get_c2(c);
    // Load particle quantities
    Interpolator2D<spline_t> interp;
    auto flag = ptc.flag[idx];
    int sp = get_ptc_type(flag);
    auto x1 = ptc.x1[idx], x2 = ptc.x2[idx];
    auto p1 = ptc.p1[idx], p2 = ptc.p2[idx], p3 = ptc.p3[idx],
         gamma = ptc.E[idx];

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
          interp(data.E1, x1, x2, c1, c2, Stagger(0b110)) * q_over_m;
      Scalar E2 =
          interp(data.E2, x1, x2, c1, c2, Stagger(0b101)) * q_over_m;
      Scalar E3 =
          interp(data.E3, x1, x2, c1, c2, Stagger(0b011)) * q_over_m;
      Scalar B1 =
          interp(data.B1, x1, x2, c1, c2, Stagger(0b001)) * q_over_m;
      Scalar B2 =
          interp(data.B2, x1, x2, c1, c2, Stagger(0b010)) * q_over_m;
      Scalar B3 =
          interp(data.B3, x1, x2, c1, c2, Stagger(0b100)) * q_over_m;

      vay_push(p1, p2, p3, gamma, E1, E2, E3, B1, B2, B3, q_over_m, dt);

      // if (dev_params.rad_cooling_on && sp != (int)ParticleType::ion)
      // {
      //   sync_kill_perp(p1, p2, p3, gamma, B1, B2, B3, E1, E2, E3,
      //                  q_over_m);
      // }

      ptc.p1[idx] = p1;
      ptc.p2[idx] = p2;
      ptc.p3[idx] = p3;
      ptc.E[idx] = gamma;
    }
  }
}

__global__ void
ptc_push_cart_3d(data_ptrs data, size_t num, Scalar dt) {
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num;
       idx += blockDim.x * gridDim.x) {
    auto &ptc = data.particles;

    auto c = ptc.cell[idx];
    // Skip empty particles
    if (c == MAX_CELL) continue;
    if (!dev_mesh.is_in_bulk(c)) {
      ptc.cell[idx] = MAX_CELL;
      continue;
    }
    int c1 = dev_mesh.get_c1(c);
    int c2 = dev_mesh.get_c2(c);
    int c3 = dev_mesh.get_c3(c);
    // Load particle quantities
    Interpolator3D<spline_t> interp;
    auto flag = ptc.flag[idx];
    int sp = get_ptc_type(flag);
    auto x1 = ptc.x1[idx], x2 = ptc.x2[idx], x3 = ptc.x3[idx];
    auto p1 = ptc.p1[idx], p2 = ptc.p2[idx], p3 = ptc.p3[idx],
         gamma = ptc.E[idx];

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
    size_t global_offset = data.E1.compute_offset(c1, c2, c3);
    if (!check_bit(flag, ParticleFlag::ignore_EM)) {
      Scalar E1 =
          interp(data.E1, x1, x2, x3, c1, c2, c3, Stagger(0b110)) *
          // interpolate(data.E1, global_offset, {x1, x2, x3},
          //             Stagger(0b110), data.E1.p.pitch,
          //             data.E1.p.ysize) *
          q_over_m;
      Scalar E2 =
          interp(data.E2, x1, x2, x3, c1, c2, c3, Stagger(0b101)) *
          q_over_m;
      Scalar E3 =
          interp(data.E3, x1, x2, x3, c1, c2, c3, Stagger(0b011)) *
          q_over_m;
      Scalar B1 =
          interp(data.B1, x1, x2, x3, c1, c2, c3, Stagger(0b001)) *
          q_over_m;
      Scalar B2 =
          interp(data.B2, x1, x2, x3, c1, c2, c3, Stagger(0b010)) *
          q_over_m;
      Scalar B3 =
          interp(data.B3, x1, x2, x3, c1, c2, c3, Stagger(0b100)) *
          q_over_m;

      vay_push(p1, p2, p3, gamma, E1, E2, E3, B1, B2, B3, q_over_m, dt);

      // if (dev_params.rad_cooling_on && sp != (int)ParticleType::ion)
      // {
      //   sync_kill_perp(p1, p2, p3, gamma, B1, B2, B3, E1, E2, E3,
      //                  q_over_m);
      // }

      ptc.p1[idx] = p1;
      ptc.p2[idx] = p2;
      ptc.p3[idx] = p3;
      ptc.E[idx] = gamma;
    }
  }
}

__global__ void
deposit_current_cart_1d(data_ptrs data, size_t num, Scalar dt,
                        uint32_t step) {
  auto &ptc = data.particles;
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num;
       idx += blockDim.x * gridDim.x) {
    auto c = ptc.cell[idx];
    // Skip empty particles
    if (c == MAX_CELL || idx >= num) continue;

    // Load particle quantities
    spline_t interp;
    auto v1 = ptc.p1[idx], v2 = ptc.p2[idx], v3 = ptc.p3[idx];
    Scalar gamma = ptc.E[idx];

    auto flag = ptc.flag[idx];
    int sp = get_ptc_type(flag);
    auto w = ptc.weight[idx];
    auto old_x1 = ptc.x1[idx], old_x2 = ptc.x2[idx],
         old_x3 = ptc.x3[idx];

    Scalar x = dev_mesh.pos(0, c, old_x1);
    Scalar y = old_x2;
    Scalar z = old_x3;

    v1 = v1 / gamma;
    v2 = v2 / gamma;
    v3 = v3 / gamma;

    Pos_t new_x1 = old_x1 + (v1 * dt) / dev_mesh.delta[0];

    int dc1 = floor(new_x1);
#ifndef NDEBUG
    if (dc1 > 1 || dc1 < -1)
      printf("----------------- Error: moved more than 1 cell!");
#endif
    new_x1 -= (Pos_t)dc1;

    ptc.cell[idx] = c + dc1;

    ptc.x1[idx] = new_x1;
    ptc.x2[idx] = y + v2 * dt;
    ptc.x3[idx] = z + v3 * dt;

    // step 2: Deposit current
    if (check_bit(flag, ParticleFlag::ignore_current)) continue;
    Scalar weight = dev_charges[sp] * w;

    int i_0 = (dc1 == -1 ? -spline_t::radius - 1 : -spline_t::radius);
    int i_1 = (dc1 == 1 ? spline_t::radius : spline_t::radius - 1);
    Scalar djx = 0.0f;
    for (int i = i_0; i <= i_1; i++) {
      Scalar sx0 = interp(-old_x1 + i + 1);
      Scalar sx1 = interp(-new_x1 + (i + 1 - dc1));

      // j1 is movement in x1
      int offset = (i + c) * sizeof(Scalar);
      djx += sx1 - sx0;
      atomicAdd(&data.J1[offset + sizeof(Scalar)], -weight * djx);

      // j2 is simply v2 times rho at center
      Scalar val1 = 0.5 * (sx0 + sx1);
      atomicAdd(&data.J2[offset], weight * v2 * val1);

      // j3 is simply v3 times rho at center
      atomicAdd(&data.J3[offset], weight * v3 * val1);

      // rho is deposited at the final position
      if ((step + 1) % dev_params.data_interval == 0) {
        atomicAdd(&data.Rho[sp][offset], weight * sx1);
      }
    }
  }
}

__global__ void
deposit_current_cart_2d(data_ptrs data, size_t num, Scalar dt,
                        uint32_t step) {
  auto &ptc = data.particles;
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num;
       idx += blockDim.x * gridDim.x) {
    auto c = ptc.cell[idx];
    // Skip empty particles
    if (c == MAX_CELL || idx >= num) continue;

    // Load particle quantities
    spline_t interp;
    int c1 = dev_mesh.get_c1(c);
    int c2 = dev_mesh.get_c2(c);
    auto v1 = ptc.p1[idx], v2 = ptc.p2[idx], v3 = ptc.p3[idx];
    Scalar gamma = ptc.E[idx];

    auto flag = ptc.flag[idx];
    int sp = get_ptc_type(flag);
    auto w = ptc.weight[idx];
    auto old_x1 = ptc.x1[idx], old_x2 = ptc.x2[idx],
         old_x3 = ptc.x3[idx];

    Scalar x = dev_mesh.pos(0, c1, old_x1);
    Scalar y = dev_mesh.pos(1, c2, old_x2);

    v1 = v1 / gamma;
    v2 = v2 / gamma;
    v3 = v3 / gamma;

    Pos_t new_x1 = old_x1 + (v1 * dt) * dev_mesh.inv_delta[0];
    Pos_t new_x2 = old_x2 + (v2 * dt) * dev_mesh.inv_delta[1];

    int dc1 = floor(new_x1);
    int dc2 = floor(new_x2);
#ifndef NDEBUG
    if (dc1 > 1 || dc1 < -1 || dc2 > 1 || dc2 < -1)
      printf("----------------- Error: moved more than 1 cell!");
#endif
    new_x1 -= (Pos_t)dc1;
    new_x2 -= (Pos_t)dc2;

    ptc.cell[idx] = dev_mesh.get_idx(c1 + dc1, c2 + dc2);

    ptc.x1[idx] = new_x1;
    ptc.x2[idx] = new_x2;
    ptc.x3[idx] = old_x3 + v3 * dt;

    // step 2: Deposit current
    if (check_bit(flag, ParticleFlag::ignore_current)) continue;
    Scalar weight = -dev_charges[sp] * w;

    // int j_0 = (dc2 == -1 ? -2 : -1);
    // int j_1 = (dc2 == 1 ? 1 : 0);
    int j_0 = (dc2 == -1 ? -spline_t::radius - 1 : -spline_t::radius);
    int j_1 = (dc2 == 1 ? spline_t::radius : spline_t::radius - 1);
    int i_0 = (dc1 == -1 ? -spline_t::radius - 1 : -spline_t::radius);
    int i_1 = (dc1 == 1 ? spline_t::radius : spline_t::radius - 1);
    Scalar djy[spline_t::support + 1] = {0.0f};
    for (int j = j_0; j <= j_1; j++) {
      Scalar sy0 = interp(-old_x2 + j + 1);
      Scalar sy1 = interp(-new_x2 + (j + 1 - dc2));

      size_t j_offset = (j + c2) * data.J1.p.pitch;
      Scalar djx = 0.0f;
      for (int i = i_0; i <= i_1; i++) {
        Scalar sx0 = interp(-old_x1 + i + 1);
        Scalar sx1 = interp(-new_x1 + (i + 1 - dc1));

        // j1 is movement in x1
        int offset = j_offset + (i + c1) * sizeof(Scalar);
        djx += movement2d(sy0, sy1, sx0, sx1);
        atomicAdd(&data.J1[offset + sizeof(Scalar)], weight * djx);

        // j2 is movement in x2
        djy[i - i_0] += movement2d(sx0, sx1, sy0, sy1);
        atomicAdd(&data.J2[offset + data.J2.p.pitch],
                  weight * djy[i - i_0]);

        // j3 is simply v3 times rho at volume average
        Scalar val2 = center2d(sx0, sx1, sy0, sy1);
        atomicAdd(&data.J3[offset], -weight * v3 * val2);

        // rho is deposited at the final position
        if ((step + 1) % dev_params.data_interval == 0) {
          Scalar s1 = sx1 * sy1;
          atomicAdd(&data.Rho[sp][offset], -weight * s1);
        }
      }
    }
  }
}

__global__ void
deposit_current_cart_3d(data_ptrs data, size_t num, Scalar dt,
                        uint32_t step) {
  auto &ptc = data.particles;
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num;
       idx += blockDim.x * gridDim.x) {
    auto c = ptc.cell[idx];
    // Skip empty particles
    if (c == MAX_CELL || idx >= num) continue;

    // Load particle quantities
    spline_t interp;
    int c1 = dev_mesh.get_c1(c);
    int c2 = dev_mesh.get_c2(c);
    int c3 = dev_mesh.get_c3(c);
    auto v1 = ptc.p1[idx], v2 = ptc.p2[idx], v3 = ptc.p3[idx];
    Scalar gamma = ptc.E[idx];

    auto flag = ptc.flag[idx];
    int sp = get_ptc_type(flag);
    auto w = ptc.weight[idx];
    auto old_x1 = ptc.x1[idx], old_x2 = ptc.x2[idx],
         old_x3 = ptc.x3[idx];

    Scalar x = dev_mesh.pos(0, c1, old_x1);
    Scalar y = dev_mesh.pos(1, c2, old_x2);
    Scalar z = dev_mesh.pos(2, c3, old_x3);

    v1 = v1 / gamma;
    v2 = v2 / gamma;
    v3 = v3 / gamma;

    Pos_t new_x1 = old_x1 + (v1 * dt) * dev_mesh.inv_delta[0];
    Pos_t new_x2 = old_x2 + (v2 * dt) * dev_mesh.inv_delta[1];
    Pos_t new_x3 = old_x3 + (v3 * dt) * dev_mesh.inv_delta[2];

    int dc1 = floor(new_x1);
    int dc2 = floor(new_x2);
    int dc3 = floor(new_x3);
#ifndef NDEBUG
    if (dc1 > 1 || dc1 < -1 || dc2 > 1 || dc2 < -1 || dc3 > 1 ||
        dc3 < -1)
      printf("----------------- Error: moved more than 1 cell!");
#endif
    new_x1 -= (Pos_t)dc1;
    new_x2 -= (Pos_t)dc2;
    new_x3 -= (Pos_t)dc3;

    ptc.cell[idx] = dev_mesh.get_idx(c1 + dc1, c2 + dc2, c3 + dc3);

    ptc.x1[idx] = new_x1;
    ptc.x2[idx] = new_x2;
    ptc.x3[idx] = new_x3;

    // step 2: Deposit current
    if (check_bit(flag, ParticleFlag::ignore_current)) continue;
    Scalar weight = -dev_charges[sp] * w;

    int k_0 = (dc3 == -1 ? -spline_t::radius - 1 : -spline_t::radius);
    int k_1 = (dc3 == 1 ? spline_t::radius : spline_t::radius - 1);
    int j_0 = (dc2 == -1 ? -spline_t::radius - 1 : -spline_t::radius);
    int j_1 = (dc2 == 1 ? spline_t::radius : spline_t::radius - 1);
    int i_0 = (dc1 == -1 ? -spline_t::radius - 1 : -spline_t::radius);
    int i_1 = (dc1 == 1 ? spline_t::radius : spline_t::radius - 1);
    // Zero initialize both arrays
    Scalar djz[spline_t::support + 1][spline_t::support + 1] = {0.0f};
    for (int k = k_0; k <= k_1; k++) {
      Scalar sz0 = interp(-old_x3 + k + 1);
      Scalar sz1 = interp(-new_x3 + (k + 1 - dc3));

      size_t k_offset = (k + c3) * data.J1.p.pitch * data.J1.p.ysize;
      Scalar djy[spline_t::support + 1] = {0.0f};
      for (int j = j_0; j <= j_1; j++) {
        Scalar sy0 = interp(-old_x2 + j + 1);
        Scalar sy1 = interp(-new_x2 + (j + 1 - dc2));

        size_t j_offset = (j + c2) * data.J1.p.pitch;
        Scalar djx = 0.0f;
        for (int i = i_0; i <= i_1; i++) {
          Scalar sx0 = interp(-old_x1 + i + 1);
          Scalar sx1 = interp(-new_x1 + (i + 1 - dc1));

          // j1 is movement in x1
          int offset = k_offset + j_offset + (i + c1) * sizeof(Scalar);
          djx += movement3d(sy0, sy1, sz0, sz1, sx0, sx1);
          atomicAdd(&data.J1[offset + sizeof(Scalar)], weight * djx);

          // j2 is movement in x2
          djy[i - i_0] += movement3d(sx0, sx1, sz0, sz1, sy0, sy1);
          atomicAdd(&data.J2[offset + data.J2.p.pitch],
                    weight * djy[i - i_0]);

          // j3 is simply v3 times rho at volume average
          djz[j - j_0][i - i_0] +=
              movement3d(sx0, sx1, sy0, sy1, sz0, sz1);
          atomicAdd(
              &data.J3[offset + data.J3.p.pitch * data.J3.p.ysize],
              weight * djz[j - j_0][i - i_0]);

          // rho is deposited at the final position
          if ((step + 1) % dev_params.data_interval == 0) {
            Scalar s1 = sx1 * sy1 * sz1;
            atomicAdd(&data.Rho[sp][offset], -weight * s1);
          }
        }
      }
    }
  }
}

__global__ void
filter_current_cart_1d(pitchptr<Scalar> j, pitchptr<Scalar> j_tmp,
                       bool boundary_lower0, bool boundary_upper0) {
  // Load position parameters
  int n1 = dev_mesh.guard[0] + blockIdx.x * blockDim.x + threadIdx.x;
  // size_t globalOffset = n2 * j.p.pitch + n1 * sizeof(Scalar);
  size_t globalOffset = j.compute_offset(n1);

  size_t d1_plus = sizeof(Scalar);
  if (boundary_upper0 && n1 == dev_mesh.dims[0] - dev_mesh.guard[0] - 1)
    d1_plus = 0;
  // (n1 < dev_mesh.dims[0] - dev_mesh.guard[0] - 1 ? sizeof(Scalar)
  //                                                : 0);
  size_t d1_minus = -sizeof(Scalar);
  if (boundary_lower0 && n1 == dev_mesh.guard[0]) d1_minus = 0;
  // (n1 > dev_mesh.guard[0] ? sizeof(Scalar) : 0);
  // (n2 > dev_mesh.guard[1] ? j.pitch : 0);
  // Do the actual computation here
  j_tmp[globalOffset] = 0.5f * j[globalOffset];
  j_tmp[globalOffset] += 0.25f * j[globalOffset + d1_plus];
  j_tmp[globalOffset] += 0.25f * j[globalOffset + d1_minus];
}

__global__ void
filter_current_cart_2d(pitchptr<Scalar> j, pitchptr<Scalar> j_tmp,
                       bool boundary_lower0, bool boundary_upper0,
                       bool boundary_lower1, bool boundary_upper1) {
  // Load position parameters
  int n1 = dev_mesh.guard[0] + blockIdx.x * blockDim.x + threadIdx.x;
  int n2 = dev_mesh.guard[1] + blockIdx.y * blockDim.y + threadIdx.y;
  // size_t globalOffset = n2 * j.p.pitch + n1 * sizeof(Scalar);
  size_t globalOffset = j.compute_offset(n1, n2);

  size_t d1_plus = sizeof(Scalar);
  if (boundary_upper0 && n1 == dev_mesh.dims[0] - dev_mesh.guard[0] - 1)
    d1_plus = 0;
  // (n1 < dev_mesh.dims[0] - dev_mesh.guard[0] - 1 ? sizeof(Scalar)
  //                                                : 0);
  size_t d1_minus = sizeof(Scalar);
  if (boundary_lower0 && n1 == dev_mesh.guard[0]) d1_minus = 0;
  // (n1 > dev_mesh.guard[0] ? sizeof(Scalar) : 0);
  size_t d2_plus = j.p.pitch;
  if (boundary_upper1 && n2 == dev_mesh.dims[1] - dev_mesh.guard[1] - 1)
    d2_plus = 0;
  // (n2 < dev_mesh.dims[1] - dev_mesh.guard[1] - 1 ? j.pitch : 0);
  size_t d2_minus = j.p.pitch;
  if (boundary_lower1 && n2 == dev_mesh.guard[1]) d2_minus = 0;
  // (n2 > dev_mesh.guard[1] ? j.pitch : 0);
  // Do the actual computation here
  j_tmp[globalOffset] = 0.25f * j[globalOffset];
  j_tmp[globalOffset] += 0.125f * j[globalOffset + d1_plus];
  j_tmp[globalOffset] += 0.125f * j[globalOffset - d1_minus];
  j_tmp[globalOffset] += 0.125f * j[globalOffset + d2_plus];
  j_tmp[globalOffset] += 0.125f * j[globalOffset - d2_minus];
  j_tmp[globalOffset] += 0.0625f * j[globalOffset + d1_plus + d2_plus];
  j_tmp[globalOffset] += 0.0625f * j[globalOffset - d1_minus + d2_plus];
  j_tmp[globalOffset] += 0.0625f * j[globalOffset + d1_plus - d2_minus];
  j_tmp[globalOffset] +=
      0.0625f * j[globalOffset - d1_minus - d2_minus];
}

__global__ void
filter_current_cart_3d(pitchptr<Scalar> j, pitchptr<Scalar> j_tmp,
                       bool boundary_lower0, bool boundary_upper0,
                       bool boundary_lower1, bool boundary_upper1,
                       bool boundary_lower2, bool boundary_upper2) {
  // Load position parameters
  int n1 = dev_mesh.guard[0] + blockIdx.x * blockDim.x + threadIdx.x;
  int n2 = dev_mesh.guard[1] + blockIdx.y * blockDim.y + threadIdx.y;
  int n3 = dev_mesh.guard[2] + blockIdx.z * blockDim.z + threadIdx.z;
  // size_t globalOffset = n2 * j.p.pitch + n1 * sizeof(Scalar);
  size_t globalOffset = j.compute_offset(n1, n2, n3);

  size_t d1_plus = sizeof(Scalar);
  if (boundary_upper0 && n1 == dev_mesh.dims[0] - dev_mesh.guard[0] - 1)
    d1_plus = 0;
  // (n1 < dev_mesh.dims[0] - dev_mesh.guard[0] - 1 ? sizeof(Scalar)
  //                                                : 0);
  size_t d1_minus = sizeof(Scalar);
  if (boundary_lower0 && n1 == dev_mesh.guard[0]) d1_minus = 0;
  // (n1 > dev_mesh.guard[0] ? sizeof(Scalar) : 0);
  size_t d2_plus = j.p.pitch;
  if (boundary_upper1 && n2 == dev_mesh.dims[1] - dev_mesh.guard[1] - 1)
    d2_plus = 0;
  // (n2 < dev_mesh.dims[1] - dev_mesh.guard[1] - 1 ? j.pitch : 0);
  size_t d2_minus = j.p.pitch;
  if (boundary_lower1 && n2 == dev_mesh.guard[1]) d2_minus = 0;
  // (n2 > dev_mesh.guard[1] ? j.pitch : 0);
  size_t d3[2] = {j.p.pitch * j.p.ysize, -j.p.pitch * j.p.ysize};
  if (boundary_upper2 && n3 == dev_mesh.dims[2] - dev_mesh.guard[2] - 1)
    d3[0] = 0;
  // (n2 < dev_mesh.dims[1] - dev_mesh.guard[1] - 1 ? j.pitch : 0);
  if (boundary_lower2 && n3 == dev_mesh.guard[2]) d3[1] = 0;
  // (n2 > dev_mesh.guard[1] ? j.pitch : 0);
  // Do the actual computation here
  j_tmp[globalOffset] = 0.125f * j[globalOffset];
  j_tmp[globalOffset] += 0.0625f * j[globalOffset + d1_plus];
  j_tmp[globalOffset] += 0.0625f * j[globalOffset - d1_minus];
  j_tmp[globalOffset] += 0.0625f * j[globalOffset + d2_plus];
  j_tmp[globalOffset] += 0.0625f * j[globalOffset - d2_minus];
  j_tmp[globalOffset] += 0.03125f * j[globalOffset + d1_plus + d2_plus];
  j_tmp[globalOffset] +=
      0.03125f * j[globalOffset - d1_minus + d2_plus];
  j_tmp[globalOffset] +=
      0.03125f * j[globalOffset + d1_plus - d2_minus];
  j_tmp[globalOffset] +=
      0.03125f * j[globalOffset - d1_minus - d2_minus];
  for (int i = 0; i < 2; i++) {
    j_tmp[globalOffset] += 0.0625f * j[globalOffset + d3[i]];
    j_tmp[globalOffset] += 0.03152f * j[globalOffset + d1_plus + d3[i]];
    j_tmp[globalOffset] +=
        0.03152f * j[globalOffset - d1_minus + d3[i]];
    j_tmp[globalOffset] += 0.03152f * j[globalOffset + d2_plus + d3[i]];
    j_tmp[globalOffset] +=
        0.03152f * j[globalOffset - d2_minus + d3[i]];
    j_tmp[globalOffset] +=
        0.015625f * j[globalOffset + d1_plus + d2_plus + d3[i]];
    j_tmp[globalOffset] +=
        0.015625f * j[globalOffset - d1_minus + d2_plus + d3[i]];
    j_tmp[globalOffset] +=
        0.015625f * j[globalOffset + d1_plus - d2_minus + d3[i]];
    j_tmp[globalOffset] +=
        0.015625f * j[globalOffset - d1_minus - d2_minus + d3[i]];
  }
}

__global__ void
move_photons_cart_1d(photon_data photons, size_t num, Scalar dt) {
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num;
       idx += blockDim.x * gridDim.x) {
    auto c = photons.cell[idx];
    // Skip empty particles
    if (c == MAX_CELL) continue;
    // Load particle quantities
    auto p1 = photons.p1[idx], p2 = photons.p2[idx],
         p3 = photons.p3[idx];
    Scalar E = std::sqrt(p1 * p1 + p2 * p2 + p3 * p3);

    auto old_x1 = photons.x1[idx], old_x2 = photons.x2[idx],
         old_x3 = photons.x3[idx];

    Pos_t new_x1 = old_x1 + p1 / E * dt * dev_mesh.inv_delta[0];
    // printf("new_x1 is %f, new_x2 is %f, old_x1 is %f, old_x2 is
    // %f\n", new_x1, new_x2, old_x1, old_x2);
    int dc1 = floor(new_x1);
    photons.cell[idx] = c + dc1;
    new_x1 -= (Pos_t)dc1;

    // printf("new_x1 is %f, new_x2 is %f, dc2 = %d\n", new_x1, new_x2,
    // dc2);
    photons.x1[idx] = new_x1;
    photons.x2[idx] = old_x2 + p2 * dt / E;
    photons.x3[idx] = old_x3 + p3 * dt / E;
    photons.path_left[idx] -= dt;
  }
}

__global__ void
move_photons_cart_2d(photon_data photons, size_t num, Scalar dt) {
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
    v1 /= E;
    v2 /= E;
    v3 /= E;

    auto old_x1 = photons.x1[idx], old_x2 = photons.x2[idx],
         old_x3 = photons.x3[idx];

    Pos_t new_x1 = old_x1 + v1 * dt * dev_mesh.inv_delta[0];
    Pos_t new_x2 = old_x2 + v2 * dt * dev_mesh.inv_delta[1];
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
    photons.x3[idx] = old_x3 + v3 * dt;
    photons.path_left[idx] -= dt;
  }
}

__global__ void
move_photons_cart_3d(photon_data photons, size_t num, Scalar dt) {
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num;
       idx += blockDim.x * gridDim.x) {
    auto c = photons.cell[idx];
    // Skip empty particles
    if (c == MAX_CELL) continue;
    // Load particle quantities
    int c1 = dev_mesh.get_c1(c);
    int c2 = dev_mesh.get_c2(c);
    int c3 = dev_mesh.get_c3(c);
    auto v1 = photons.p1[idx], v2 = photons.p2[idx],
         v3 = photons.p3[idx];
    Scalar E = std::sqrt(v1 * v1 + v2 * v2 + v3 * v3);
    v1 /= E;
    v2 /= E;
    v3 /= E;

    auto old_x1 = photons.x1[idx], old_x2 = photons.x2[idx],
         old_x3 = photons.x3[idx];

    Pos_t new_x1 = old_x1 + v1 * dt * dev_mesh.inv_delta[0];
    Pos_t new_x2 = old_x2 + v2 * dt * dev_mesh.inv_delta[1];
    Pos_t new_x3 = old_x3 + v3 * dt * dev_mesh.inv_delta[2];
    // printf("new_x1 is %f, new_x2 is %f, old_x1 is %f, old_x2 is
    // %f\n", new_x1, new_x2, old_x1, old_x2);
    int dc1 = floor(new_x1);
    int dc2 = floor(new_x2);
    int dc3 = floor(new_x3);
    photons.cell[idx] = dev_mesh.get_idx(c1 + dc1, c2 + dc2, c3 + dc3);
    new_x1 -= (Pos_t)dc1;
    new_x2 -= (Pos_t)dc2;
    new_x3 -= (Pos_t)dc3;

    // printf("new_x1 is %f, new_x2 is %f, dc2 = %d\n", new_x1, new_x2,
    // dc2);
    photons.x1[idx] = new_x1;
    photons.x2[idx] = new_x2;
    photons.x3[idx] = new_x3;
    photons.path_left[idx] -= dt;
  }
}

}  // namespace Kernels

ptc_updater::ptc_updater(sim_environment &env) : m_env(env) {
  m_tmp_j = multi_array<Scalar>(env.local_grid().extent());
}

ptc_updater::~ptc_updater() {}

void
ptc_updater::update_particles(sim_data &data, double dt,
                              uint32_t step) {
  auto data_p = get_data_ptrs(data);
  data.J.initialize();
  for (int i = 0; i < data.env.params().num_species; i++) {
    data.Rho[i].initialize();
  }
  auto &grid = m_env.local_grid();

  timer::stamp("ptc_update");
  // Skip empty particle array
  if (data.particles.number() > 0) {
    Logger::print_info("Updating {} particles in Cartesian coordinates",
                       data.particles.number());

    timer::stamp("ptc_push");
    // Push the particle with E and B fields
    if (grid.dim() == 1) {
      Kernels::ptc_push_cart_1d<<<256, 512>>>(
          data_p, data.particles.number(), dt);
    } else if (grid.dim() == 2) {
      Kernels::ptc_push_cart_2d<<<256, 512>>>(
          data_p, data.particles.number(), dt);
    } else if (grid.dim() == 3) {
      Kernels::ptc_push_cart_3d<<<256, 512>>>(
          data_p, data.particles.number(), dt);
    }
    CudaCheckError();
    CudaSafeCall(cudaDeviceSynchronize());

    timer::show_duration_since_stamp("Pushing particles", "ms",
                                     "ptc_push");

    timer::stamp("ptc_deposit");
    // Deposit current and move the particle
    if (grid.dim() == 1) {
      Kernels::deposit_current_cart_1d<<<256, 512>>>(
          data_p, data.particles.number(), dt, step);
    } else if (grid.dim() == 2) {
      Kernels::deposit_current_cart_2d<<<256, 512>>>(
          data_p, data.particles.number(), dt, step);
    } else if (grid.dim() == 3) {
      Kernels::deposit_current_cart_3d<<<256, 512>>>(
          data_p, data.particles.number(), dt, step);
    }
    CudaCheckError();
    CudaSafeCall(cudaDeviceSynchronize());
    timer::show_duration_since_stamp("Depositing particles", "ms",
                                     "ptc_deposit");

    m_env.send_add_guard_cells(data.J);
    if ((step + 1) % data.env.params().data_interval == 0) {
      for (int i = 0; i < data.env.params().num_species; i++) {
        m_env.send_add_guard_cells(data.Rho[i]);
      }
    }

    smooth_current(data, step);
    m_env.send_particles(data.particles);
  }
  // timer::show_duration_since_stamp("Sending guard cells", "ms",
  // "comm");
  if (data.photons.number() > 0) {
    timer::stamp("photon_move");
    Logger::print_info("Updating {} photons in Cartesian coordinates",
                       data.photons.number());

    // Push the particle with E and B fields
    if (grid.dim() == 1) {
      Kernels::move_photons_cart_1d<<<256, 512>>>(
          data.photons.data(), data.photons.number(), dt);
    } else if (grid.dim() == 2) {
      Kernels::move_photons_cart_2d<<<256, 512>>>(
          data.photons.data(), data.photons.number(), dt);
    } else if (grid.dim() == 3) {
      Kernels::move_photons_cart_3d<<<256, 512>>>(
          data.photons.data(), data.photons.number(), dt);
    }
    CudaCheckError();
    CudaSafeCall(cudaDeviceSynchronize());
    m_env.send_particles(data.photons);
    timer::show_duration_since_stamp("Moving photons", "ms",
                                     "photon_move");
  }
  apply_boundary(data, dt, step);
  timer::show_duration_since_stamp("Ptc update", "ms", "ptc_update");
}

void
ptc_updater::apply_boundary(sim_data &data, double dt, uint32_t step) {
  // data.particles.clear_guard_cells(m_env.local_grid());
  // data.photons.clear_guard_cells(m_env.local_grid());
  // CudaSafeCall(cudaDeviceSynchronize());

  // TODO: apply other boundary conditions on current/rho/particles
}

void
ptc_updater::smooth_current(sim_data &data, uint32_t step) {
  Logger::print_debug("current smoothing {} times",
                      m_env.params().current_smoothing);
  auto &grid = m_env.local_grid();
  auto &mesh = grid.mesh();

  for (int i = 0; i < m_env.params().current_smoothing; i++) {
    smooth_current(data.J.data(0));
    smooth_current(data.J.data(1));
    smooth_current(data.J.data(2));

    m_env.send_guard_cells(data.J);
    if ((step + 1) % data.env.params().data_interval == 0) {
      for (int i = 0; i < data.env.params().num_species; i++) {
        smooth_current(data.Rho[i].data());
        m_env.send_guard_cells(data.Rho[i]);
      }
    }
    CudaSafeCall(cudaDeviceSynchronize());
  }
}

void
ptc_updater::smooth_current(multi_array<Scalar> &array) {
  auto &grid = m_env.local_grid();
  auto &mesh = grid.mesh();
  if (grid.dim() == 1) {
    dim3 blockSize(512);
    dim3 gridSize((mesh.reduced_dim(0) + blockSize.x - 1) /
                  blockSize.x);
    Kernels::filter_current_cart_1d<<<gridSize, blockSize>>>(
        get_pitchptr(array), get_pitchptr(m_tmp_j),
        m_env.is_boundary(0), m_env.is_boundary(1));
  } else if (grid.dim() == 2) {
    dim3 blockSize(32, 16);
    dim3 gridSize(
        (mesh.reduced_dim(0) + blockSize.x - 1) / blockSize.x,
        (mesh.reduced_dim(1) + blockSize.y - 1) / blockSize.y);

    Kernels::filter_current_cart_2d<<<gridSize, blockSize>>>(
        get_pitchptr(array), get_pitchptr(m_tmp_j),
        m_env.is_boundary(0), m_env.is_boundary(1),
        m_env.is_boundary(2), m_env.is_boundary(3));
  } else if (grid.dim() == 3) {
    dim3 blockSize(32, 8, 4);
    dim3 gridSize(
        (mesh.reduced_dim(0) + blockSize.x - 1) / blockSize.x,
        (mesh.reduced_dim(1) + blockSize.y - 1) / blockSize.y,
        (mesh.reduced_dim(2) + blockSize.z - 1) / blockSize.z);

    Kernels::filter_current_cart_3d<<<gridSize, blockSize>>>(
        get_pitchptr(array), get_pitchptr(m_tmp_j),
        m_env.is_boundary(0), m_env.is_boundary(1),
        m_env.is_boundary(2), m_env.is_boundary(3),
        m_env.is_boundary(4), m_env.is_boundary(5));
  }

  CudaCheckError();
  array.copy_from(m_tmp_j);
}

}  // namespace Aperture
