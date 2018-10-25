#include "cuda/constant_mem.h"
#include "cuda/cudaUtility.h"
#include "cuda/kernels.h"
#include "data/detail/multi_array_utils.hpp"
#include "ptc_updater.h"
#include "sim_data.h"
#include "sim_environment.h"
#include "utils/interpolation.cuh"
#include "utils/logger.h"
#include "utils/util_functions.h"

namespace Aperture {

namespace Kernels {

typedef Spline::cloud_in_cell spline_t;

HD_INLINE Scalar
cloud_in_cell_f(Scalar dx) {
  return max(1.0 - std::abs(dx), 0.0);
}

HD_INLINE Scalar
interpolate(Scalar rel_pos, int ptc_cell, int target_cell) {
  Scalar dx =
      ((Scalar)target_cell + 0.5 - (rel_pos + Scalar(ptc_cell)));
  return cloud_in_cell_f(dx);
}

HD_INLINE Scalar
center2d(Scalar sx0, Scalar sx1, Scalar sy0, Scalar sy1) {
  return (2.0f * sx1 * sy1 + sx0 * sy1 + sx1 * sy0 + 2.0f * sx0 * sy0) *
         0.1666667f;
}

HD_INLINE Scalar
movement3d(Scalar sx0, Scalar sx1, Scalar sy0, Scalar sy1, Scalar sz0,
           Scalar sz1) {
  return (sz1 - sz0) * center2d(sx0, sx1, sy0, sy1);
}

HD_INLINE Scalar
movement2d(Scalar sx0, Scalar sx1, Scalar sy0, Scalar sy1) {
  return (sy1 - sy0) * 0.5f * (sx0 + sx1);
}

#define MIN_BLOCKS_PER_MP 3
#define MAX_THREADS_PER_BLOCK 256
__global__ void
__launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP)
    update_particles(particle_data ptc, size_t num, fields_data fields,
                     Scalar dt) {
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num;
       idx += blockDim.x * gridDim.x) {
    auto c = ptc.cell[idx];
    // Skip empty particles
    if (c == MAX_CELL) continue;

    // Load particle quantities
    Interpolator3D<spline_t> interp;
    auto flag = ptc.flag[idx];
    int sp = get_ptc_type(flag);
    auto w = ptc.weight[idx];
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
    // step 2: Compute particle movement and update position
    Pos_t new_x1 = old_x1 + dt * p1 / gamma;
    Pos_t new_x2 = old_x2 + dt * p2 / gamma;
    Pos_t new_x3 = old_x3 + dt * p3 / gamma;
    int dc1 = floor(new_x1);
    int dc2 = floor(new_x2);
    int dc3 = floor(new_x3);
    new_x1 -= (Pos_t)dc1;
    new_x2 -= (Pos_t)dc2;
    new_x3 -= (Pos_t)dc3;
    ptc.cell[idx] = dev_mesh.get_idx(c1 + dc1, c2 + dc2, c3 + dc3);
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
      int kk = (((idx + k) % sup23) / (sup2 * sup2));
      Scalar sz0 =
          interp.interpolate(0.5f + kk - interp.radius() - old_x3);
      Scalar sz1 = interp.interpolate(0.5f + kk - interp.radius() +
                                      dc3 - new_x3);
      size_t k_offset = (kk - interp.radius() + c3) * fields.J1.pitch *
                        fields.J1.ysize;
      for (int j = 0; j < sup2; j++) {
        // int jj = j;
        int jj = (((idx + j) % sup23) / sup2) % sup2;
        Scalar sy0 =
            interp.interpolate(0.5f + jj - interp.radius() - old_x2);
        Scalar sy1 = interp.interpolate(0.5f + jj - interp.radius() +
                                        dc2 - new_x2);
        size_t j_offset = (jj - interp.radius() + c2) * fields.J1.pitch;
        // Scalar djx = 0.0f;
        for (int i = 0; i < sup2; i++) {
          // int ii = i;
          int ii = ((idx + i) % sup23) % sup2;
          Scalar sx0 =
              interp.interpolate(0.5f + ii - interp.radius() - old_x1);
          Scalar sx1 = interp.interpolate(0.5f + ii - interp.radius() +
                                          dc1 - new_x1);

          int offset = k_offset + j_offset + (ii + c1) * sizeof(Scalar);
          // djx -= wdt * dev_mesh.delta[0] *
          //        movement3d(sy0, sy1, sz0, sz1, sx0, sx1);
          // djy[k + interp.radius() + 1][i + interp.radius() + 1] -=
          //     wdt * dev_mesh.delta[1] *
          //     movement3d(sz0, sz1, sx0, sx1, sy0, sy1);
          // djz[i + interp.radius() + 1][j + interp.radius() + 1] -=
          //     wdt * dev_mesh.delta[2] *
          //     movement3d(sx0, sx1, sy0, sy1, sz0, sz1);
          atomicAdd((Scalar *)((char *)fields.J1.ptr + offset),
                    wdt * dev_mesh.delta[0] *
                        movement3d(sy0, sy1, sz0, sz1, sx0, sx1));
          atomicAdd((Scalar *)((char *)fields.J2.ptr + offset),
                    wdt * dev_mesh.delta[1] *
                        movement3d(sz0, sz1, sx0, sx1, sy0, sy1));
          atomicAdd((Scalar *)((char *)fields.J3.ptr + offset),
                    wdt * dev_mesh.delta[2] *
                        movement3d(sx0, sx1, sy0, sy1, sz0, sz1));
          atomicAdd((Scalar *)((char *)fields.Rho[sp].ptr + offset),
                    sx1 * sy1 * sz1 * dev_charges[sp]);
        }
      }
    }
  }
}

__global__ void
update_particles_2d(particle_data ptc, size_t num, fields_data fields,
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
      Scalar E1 =
          interp(fields.E1, old_x1, old_x2, c1, c2, Stagger(0b001)) *
          q_over_m;
      Scalar E2 =
          interp(fields.E2, old_x1, old_x2, c1, c2, Stagger(0b010)) *
          q_over_m;
      Scalar E3 =
          interp(fields.E3, old_x1, old_x2, c1, c2, Stagger(0b100)) *
          q_over_m;
      Scalar B1 =
          interp(fields.B1, old_x1, old_x2, c1, c2, Stagger(0b110)) *
          q_over_m;
      Scalar B2 =
          interp(fields.B2, old_x1, old_x2, c1, c2, Stagger(0b101)) *
          q_over_m;
      Scalar B3 =
          interp(fields.B3, old_x1, old_x2, c1, c2, Stagger(0b011)) *
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

    // step 2: Compute particle movement and update position
    Pos_t new_x1 = old_x1 + dt * p1 / gamma;
    Pos_t new_x2 = old_x2 + dt * p2 / gamma;
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

PtcUpdater::PtcUpdater(const Environment &env) : m_env(env) {
  m_extent = m_env.grid().extent();
  // FIXME: select the correct device?
  CudaSafeCall(cudaMallocManaged(
      &m_dev_fields.Rho,
      m_env.params().num_species * sizeof(cudaPitchedPtr)));
  m_fields_initialized = false;
}

PtcUpdater::~PtcUpdater() {
  // FIXME: select the correct device
  CudaSafeCall(cudaFree(m_dev_fields.Rho));
}

void
PtcUpdater::update_particles(SimData &data, double dt) {
  Logger::print_info("Updating particles");
  // Track the right fields
  if (!m_fields_initialized) {
    m_dev_fields.E1 = data.E.ptr(0);
    m_dev_fields.E2 = data.E.ptr(1);
    m_dev_fields.E3 = data.E.ptr(2);
    m_dev_fields.B1 = data.B.ptr(0);
    m_dev_fields.B2 = data.B.ptr(1);
    m_dev_fields.B3 = data.B.ptr(2);
    m_dev_fields.J1 = data.J.ptr(0);
    m_dev_fields.J2 = data.J.ptr(1);
    m_dev_fields.J3 = data.J.ptr(2);
    for (int i = 0; i < data.num_species; i++) {
      m_dev_fields.Rho[i] = data.Rho[i].ptr();
    }
    m_fields_initialized = true;
  }

  if (m_env.grid().dim() == 1) {
    Kernels::update_particles_1d<<<512, 512>>>(
        data.particles.data(), data.particles.number(),
        (const Scalar *)data.E.ptr(0).ptr, (Scalar *)data.J.ptr(0).ptr,
        (Scalar *)data.Rho[0].ptr().ptr, dt);
    CudaCheckError();
  } else if (m_env.grid().dim() == 2) {
    Kernels::update_particles_2d<<<256, 256>>>(data.particles.data(),
                                            data.particles.number(),
                                            m_dev_fields, dt);
    CudaCheckError();
  } else if (m_env.grid().dim() == 3) {
    Kernels::update_particles<<<256, 256>>>(data.particles.data(),
                                            data.particles.number(),
                                            m_dev_fields, dt);
    CudaCheckError();
  }
  cudaDeviceSynchronize();
}

void
PtcUpdater::handle_boundary(SimData &data) {
  erase_ptc_in_guard_cells(data.particles.data().cell,
                           data.particles.number());
}

}  // namespace Aperture
