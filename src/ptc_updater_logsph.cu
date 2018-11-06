#include "cuda/constant_mem.h"
#include "cuda/cudaUtility.h"
#include "cuda/kernels.h"
#include "data/detail/multi_array_utils.hpp"
#include "ptc_updater_helper.cuh"
#include "ptc_updater_logsph.h"
#include "sim_data.h"
#include "sim_environment.h"
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
    Scalar gamma = std::sqrt(1.0f + p1 * p1 + p2 * p2 + p3 * p3);
    if (!check_bit(flag, ParticleFlag::ignore_EM)) {
      Scalar E1 =
          (interp(fields.E1, old_x1, old_x2, c1, c2, Stagger(0b001)) +
           interp(dev_bg_fields.E1, old_x1, old_x2, c1, c2,
                  Stagger(0b001))) *
          q_over_m;
      Scalar E2 =
          (interp(fields.E2, old_x1, old_x2, c1, c2, Stagger(0b010)) +
           interp(dev_bg_fields.E2, old_x1, old_x2, c1, c2,
                  Stagger(0b010))) *
          q_over_m;
      Scalar E3 =
          (interp(fields.E3, old_x1, old_x2, c1, c2, Stagger(0b100)) +
           interp(dev_bg_fields.E3, old_x1, old_x2, c1, c2,
                  Stagger(0b100))) *
          q_over_m;
      Scalar B1 =
          (interp(fields.B1, old_x1, old_x2, c1, c2, Stagger(0b110)) +
           interp(dev_bg_fields.B1, old_x1, old_x2, c1, c2,
                  Stagger(0b110))) *
          q_over_m;
      Scalar B2 =
          (interp(fields.B2, old_x1, old_x2, c1, c2, Stagger(0b101)) +
           interp(dev_bg_fields.B2, old_x1, old_x2, c1, c2,
                  Stagger(0b101))) *
          q_over_m;
      Scalar B3 =
          (interp(fields.B3, old_x1, old_x2, c1, c2, Stagger(0b011)) +
           interp(dev_bg_fields.B3, old_x1, old_x2, c1, c2,
                  Stagger(0b011))) *
          q_over_m;

      // printf("B1 = %f, B2 = %f, B3 = %f\n", B1, B2, B3);
      // printf("B cell is %f\n", *ptrAddr(fields.B1, c1*sizeof(Scalar)
      // + c2*fields.B1.pitch)); printf("q over m is %f\n", q_over_m);

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
__launch_bounds__(512, 4)
    deposit_current_2d_log_sph(particle_data ptc, size_t num,
                               fields_data fields,
                               Grid_LogSph::mesh_ptrs mesh_ptrs,
                               Scalar dt) {
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num;
       idx += blockDim.x * gridDim.x) {
    auto c = ptc.cell[idx];
    // Skip empty particles
    if (c == MAX_CELL) continue;

    // Load particle quantities
    Interpolator2D<spline_t> interp;
    int c1 = dev_mesh.get_c1(c);
    int c2 = dev_mesh.get_c2(c);
    auto v1 = ptc.p1[idx], v2 = ptc.p2[idx], v3 = ptc.p3[idx];
    Scalar gamma = std::sqrt(1.0f + v1 * v1 + v2 * v2 + v3 * v3);

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
    Scalar r2 = dev_mesh.pos(1, c2, old_x2);
    Scalar x = std::exp(r1) * std::sin(r2) * std::cos(old_x3);
    Scalar y = std::exp(r1) * std::sin(r2) * std::sin(old_x3);
    Scalar z = std::exp(r1) * std::cos(r2);

    logsph2cart(v1, v2, v3, r1, r2, old_x3);
    x += v1 * dt;
    y += v2 * dt;
    z += v3 * dt;
    Scalar r1p = sqrt(x * x + y * y + z * z);
    Scalar r2p = acos(z / r1p);
    r1p = log(r1p);
    Scalar r3p = atan(y / x);
    if (x < 0.0) v1 *= -1.0;

    // printf("position is (%f, %f, %f)\n", exp(r1p), r2p, r3p);

    cart2logsph(v1, v2, v3, r1p, r2p, r3p);
    ptc.p1[idx] = v1 * gamma;
    ptc.p2[idx] = v2 * gamma;
    ptc.p3[idx] = v3 * gamma;

    // Scalar old_pos3 =
    Pos_t new_x1 = old_x1 + (r1p - r1) / dev_mesh.delta[0];
    Pos_t new_x2 = old_x2 + (r2p - r2) / dev_mesh.delta[1];
    // printf("new_x1 is %f, new_x2 is %f, old_x1 is %f, old_x2 is %f\n", new_x1, new_x2, old_x1, old_x2);
    int dc1 = floor(new_x1);
    int dc2 = floor(new_x2);
#ifndef NDEBUG
    if (dc1 > 1 || dc1 < -1 || dc2 > 1 || dc2 < -1)
      printf("----------------- Error: moved more than 1 cell!");
#endif
    ptc.cell[idx] = dev_mesh.get_idx(c1 + dc1, c2 + dc2);
    new_x1 -= (Pos_t)dc1;
    new_x2 -= (Pos_t)dc2;
    // printf("new_x1 is %f, new_x2 is %f, dc2 = %d\n", new_x1, new_x2, dc2);
    ptc.x1[idx] = new_x1;
    ptc.x2[idx] = new_x2;
    ptc.x3[idx] = r3p;

    // step 2: Deposit current
    if (check_bit(flag, ParticleFlag::ignore_current)) continue;
    // Scalar djz[spline_t::support + 1][spline_t::support + 1] =
    // {0.0f};
    Scalar wdt =
        // -dev_charges[sp] * dev_mesh.delta[0] * dev_mesh.delta[1] * w / dt;
        -dev_charges[sp] * w / dt;
    int sup2 = interp.support() + 2;
    // int sup22 = sup2 * sup2;
    Scalar djy[spline_t::support + 2] = {0.0f};
    for (int j = 0; j < sup2; j++) {
      int jj = j - interp.radius();
      // int jj = (((idx + j) % sup22) / sup2) - interp.radius();
      Scalar sy0 = interp.interpolate(0.5f - old_x2 + jj);
      Scalar sy1 = interp.interpolate(0.5f - new_x2 + (jj - dc2));
      // if (std::abs(sy0) < DEPOSIT_EPS && std::abs(sy1) <
      // DEPOSIT_EPS)
      //   continue;
      size_t j_offset = (jj + c2) * fields.J1.pitch;
      Scalar djx = 0.0f;
      for (int i = 0; i < sup2; i++) {
        int ii = i - interp.radius();
        // int ii = ((idx + i) % sup22) % sup2;
        Scalar sx0 = interp.interpolate(0.5f - old_x1 + ii);
        Scalar sx1 = interp.interpolate(0.5f - new_x1 + (ii - dc1));
        // if (std::abs(sx0) < DEPOSIT_EPS && std::abs(sx1) <
        // DEPOSIT_EPS)
        //   continue;

        int offset = j_offset + (ii + c1) * sizeof(Scalar);
        Scalar val0 = movement2d(sy0, sy1, sx0, sx1);
        // printf("dq0 = %f, ", val0);
        if (std::abs(val0) > 0.0f) {
          djx += wdt * val0 * dev_mesh.delta[0] * dev_mesh.delta[1];
          atomicAdd(ptrAddr(fields.J1, offset),
                    djx / *ptrAddr(mesh_ptrs.A1_e, offset));
        }
        Scalar val1 = movement2d(sx0, sx1, sy0, sy1);
        // printf("dq1 = %f, ", val1);
        if (std::abs(val1) > 0.0f) {
          djy[i] += wdt * val1 * dev_mesh.delta[0] * dev_mesh.delta[1];
          atomicAdd(ptrAddr(fields.J2, offset),
                    djy[i] / *ptrAddr(mesh_ptrs.A2_e, offset));
        }
        // printf("val1 = %f, djy[%d] = %f, ", val1, i, djy[i]);
        Scalar val2 = center2d(sx0, sx1, sy0, sy1);
        // printf("dq2 = %f, ", val2);
        if (std::abs(val2) > 0.0f)
          atomicAdd(ptrAddr(fields.J3, offset),
                    dev_charges[sp] * w * v3 * val2 /
                        *ptrAddr(mesh_ptrs.dV, offset));
        Scalar s1 = sx1 * sy1;
        // printf("s1 = %f\n", s1);
        if (std::abs(s1) > 0.0f)
          atomicAdd(ptrAddr(fields.Rho[sp], offset),
                    w * s1 * dev_charges[sp] /
                        *ptrAddr(mesh_ptrs.dV, offset));
      }
      // printf("\n");
    }
  }
}

}  // namespace Kernels

PtcUpdaterLogSph::PtcUpdaterLogSph(const Environment &env)
    : PtcUpdater(env) {
  const Grid_LogSph &grid =
      dynamic_cast<const Grid_LogSph &>(env.grid());
  m_mesh_ptrs = grid.get_mesh_ptrs();
}

PtcUpdaterLogSph::~PtcUpdaterLogSph() {}

void
PtcUpdaterLogSph::update_particles(SimData &data, double dt) {
  Logger::print_info(
      "Updating {} particles in log spherical coordinates",
      data.particles.number());

  initialize_dev_fields(data);

  if (m_env.grid().dim() == 2) {
    Kernels::vay_push_2d<<<256, 512>>>(data.particles.data(),
                                       data.particles.number(),
                                       m_dev_fields, dt);
    CudaCheckError();
    data.J.initialize();
    for (auto &rho : data.Rho) {
      rho.initialize();
    }
    Kernels::deposit_current_2d_log_sph<<<256, 512>>>(
        data.particles.data(), data.particles.number(), m_dev_fields,
        m_mesh_ptrs, dt);
    CudaCheckError();
  }
  cudaDeviceSynchronize();
}

void
PtcUpdaterLogSph::handle_boundary(SimData &data) {
  data.particles.clear_guard_cells();
}

}  // namespace Aperture