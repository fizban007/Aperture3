#include "core/detail/multi_array_utils.hpp"
#include "cuda/constant_mem.h"
#include "cuda/core/cu_sim_data.h"
#include "cuda/core/ptc_updater_helper.cuh"
#include "cuda/core/sim_environment_dev.h"
#include "cuda/cudaUtility.h"
#include "cuda/kernels.h"
#include "cuda/ptr_util.h"
#include "cuda/utils/interpolation.cuh"
#include "cuda/utils/iterate_devices.h"
#include "ptc_updater_logsph.h"
#include "utils/logger.h"
#include "utils/timer.h"
#include "utils/util_functions.h"

#define DEPOSIT_EPS 1.0e-10f

namespace Aperture {

namespace Kernels {

__device__ Scalar beta_phi(Scalar r, Scalar theta);

__device__ Scalar alpha_gr(Scalar r);

HD_INLINE void
cart2logsph(Scalar &v1, Scalar &v2, Scalar &v3, Scalar x1, Scalar x2,
            Scalar x3) {
  Scalar v1n = v1, v2n = v2, v3n = v3;
  Scalar c2 = cos(x2), s2 = sin(x2), c3 = cos(x3), s3 = sin(x3);
  v1 = v1n * s2 * c3 + v2n * s2 * s3 + v3n * c2;
  v2 = v1n * c2 * c3 + v2n * c2 * s3 - v3n * s2;
  v3 = -v1n * s3 + v2n * c3;
}

HD_INLINE void
logsph2cart(Scalar &v1, Scalar &v2, Scalar &v3, Scalar x1, Scalar x2,
            Scalar x3) {
  Scalar v1n = v1, v2n = v2, v3n = v3;
  Scalar c2 = cos(x2), s2 = sin(x2), c3 = cos(x3), s3 = sin(x3);
  v1 = v1n * s2 * c3 + v2n * c2 * c3 - v3n * s3;
  v2 = v1n * s2 * s3 + v2n * c2 * s3 + v3n * c3;
  v3 = v1n * c2 - v2n * s2;
}

__global__ void
vay_push_2d(particle_data ptc, size_t num,
            PtcUpdaterDev::fields_data fields, Scalar dt) {
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num;
       idx += blockDim.x * gridDim.x) {
    auto c = ptc.cell[idx];
    // Skip empty particles
    if (c == MAX_CELL || idx >= num) continue;
    int c1 = dev_mesh.get_c1(c);
    int c2 = dev_mesh.get_c2(c);
    if (!dev_mesh.is_in_bulk(c1, c2)) {
      ptc.cell[idx] = MAX_CELL;
      continue;
    }

    // Load particle quantities
    Interpolator2D<spline_t> interp;
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
    }
    // step 0: Grab E & M fields at the particle position
    gamma = std::sqrt(1.0f + p1 * p1 + p2 * p2 + p3 * p3);
    if (!check_bit(flag, ParticleFlag::ignore_EM)) {
      Scalar E1 =
          alpha_gr(r) *
          (interp(fields.E1, old_x1, old_x2, c1, c2, Stagger(0b110))) *
          // interp(dev_bg_fields.E1, old_x1, old_x2, c1, c2,
          //        Stagger(0b110))) *
          q_over_m;
      Scalar E2 =
          alpha_gr(r) *
          (interp(fields.E2, old_x1, old_x2, c1, c2, Stagger(0b101))) *
          // interp(dev_bg_fields.E2, old_x1, old_x2, c1, c2,
          //        Stagger(0b101))) *
          q_over_m;
      Scalar E3 =
          alpha_gr(r) *
          (interp(fields.E3, old_x1, old_x2, c1, c2, Stagger(0b011))) *
          // interp(dev_bg_fields.E3, old_x1, old_x2, c1, c2,
          //        Stagger(0b011))) *
          q_over_m;
      Scalar B1 =
          alpha_gr(r) *
          (interp(fields.B1, old_x1, old_x2, c1, c2, Stagger(0b001)) +
           interp(dev_bg_fields.B1, old_x1, old_x2, c1, c2,
                  Stagger(0b001))) *
          q_over_m;
      Scalar B2 =
          alpha_gr(r) *
          (interp(fields.B2, old_x1, old_x2, c1, c2, Stagger(0b010)) +
           interp(dev_bg_fields.B2, old_x1, old_x2, c1, c2,
                  Stagger(0b010))) *
          q_over_m;
      Scalar B3 =
          alpha_gr(r) *
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
      if (dev_params.gravity_on) {
        p1 -= dt * alpha_gr(r) * gamma * dev_params.gravity / (r * r);
        gamma = sqrt(1.0f + p1 * p1 + p2 * p2 + p3 * p3);
        if (gamma != gamma) {
          printf(
              "NaN detected after gravity! p1 is %f, p2 is %f, p3 is "
              "%f, gamma is "
              "%f\n",
              p1, p2, p3, gamma);
          asm("trap;");
        }
      }

      Scalar p = sqrt(p1 * p1 + p2 * p2 + p3 * p3);

      if (dev_params.rad_cooling_on && sp != (int)ParticleType::ion) {
        Scalar tmp1 = (E1 + (p2 * B3 - p3 * B2) / gamma) / q_over_m;
        Scalar tmp2 = (E2 + (p3 * B1 - p1 * B3) / gamma) / q_over_m;
        Scalar tmp3 = (E3 + (p1 * B2 - p2 * B1) / gamma) / q_over_m;
        Scalar tmp_sq = tmp1 * tmp1 + tmp2 * tmp2 + tmp3 * tmp3;
        Scalar bE = (p1 * E1 + p2 * E2 + p3 * E3) / (gamma * q_over_m);

        Scalar delta_p1 =
            dev_params.rad_cooling_coef *
            (((tmp2 * B3 - tmp3 * B2) + bE * E1) / q_over_m -
             gamma * p1 * (tmp_sq - bE * bE)) /
            square(dev_params.B0);
        Scalar delta_p2 =
            dev_params.rad_cooling_coef *
            (((tmp3 * B1 - tmp1 * B3) + bE * E2) / q_over_m -
             gamma * p2 * (tmp_sq - bE * bE)) /
            square(dev_params.B0);
        Scalar delta_p3 =
            dev_params.rad_cooling_coef *
            (((tmp1 * B2 - tmp2 * B1) + bE * E3) / q_over_m -
             gamma * p3 * (tmp_sq - bE * bE)) /
            square(dev_params.B0);
        // Scalar dp = sqrt(delta_p1 * delta_p1 + delta_p2 * delta_p2 +
        //                  delta_p3 * delta_p3);
        // if (dp < p) {
        // p1 += (dp < 10.0f * p || dp < 1e-2 ? delta_p1
        //                                    : 0.5 * p * delta_p1 /
        //                                    dp);
        // p2 += (dp < 10.0f * p || dp < 1e-2 ? delta_p2
        //                                    : 0.5 * p * delta_p2 /
        //                                    dp);
        // p3 += (dp < 10.0f * p || dp < 1e-2 ? delta_p3
        //                                    : 0.5 * p * delta_p3 /
        //                                    dp);
        p1 += delta_p1;
        p2 += delta_p2;
        p3 += delta_p3;
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
move_photons(photon_data photons, size_t num, Scalar dt, bool axis0,
             bool axis1) {
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
    Scalar r3p = atan2(y, x);

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
    // reflect around the axis
    // if (c2 + dc2 < dev_mesh.guard[1] && axis0) {
    if (dev_mesh.pos(1, c2 + dc2, new_x2) < 0.0f) {
      dc2 += 1;
      new_x2 = 1.0f - new_x2;
      // } else if (c2 + dc2 >= dev_mesh.dims[1] - dev_mesh.guard[1] &&
      //            axis1) {
    } else if (dev_mesh.pos(1, c2 + dc2, new_x2) >= CONST_PI) {
      dc2 -= 1;
      new_x2 = 1.0f - new_x2;
    }
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
                               Scalar dt, uint32_t step, bool axis0,
                               bool axis1) {
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

    Scalar r1 = dev_mesh.pos(0, c1, old_x1);
    Scalar exp_r1 = std::exp(r1);
    Scalar r2 = dev_mesh.pos(1, c2, old_x2);

    // v1 = alpha_gr(exp_r1) * v1 / gamma;
    // v2 = alpha_gr(exp_r1) * v2 / gamma;
    // v3 = alpha_gr(exp_r1) * v3 / gamma - beta_phi(exp_r1, r2);
    v1 = v1 / gamma;
    v2 = v2 / gamma;
    v3 = v3 / gamma - beta_phi(exp_r1, r2);
    // v3 = v3 / gamma;

    // step 1: Compute particle movement and update position
    // v3 -= 0.4f * dev_params.omega * dev_params.compactness *
    //       std::sin(r2) / (exp_r1 * exp_r1);
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
    Scalar r3p = atan2(y, x);
    // if (x < 0.0f) v1 *= -1.0f;

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
    // if (c2 + dc2 < dev_mesh.guard[1] && axis0) {
    if (dev_mesh.pos(1, c2 + dc2, new_x2) < 0.0f) {
      dc2 += 1;
      new_x2 = 1.0f - new_x2;
      // } else if (c2 + dc2 >= dev_mesh.dims[1] - dev_mesh.guard[1] &&
      //            axis1) {
    } else if (dev_mesh.pos(1, c2 + dc2, new_x2) >= CONST_PI) {
      dc2 -= 1;
      new_x2 = 1.0f - new_x2;
    }
    ptc.cell[idx] = dev_mesh.get_idx(c1 + dc1, c2 + dc2);
    // printf("new_x1 is %f, new_x2 is %f, dc2 = %d\n", new_x1, new_x2,
    // dc2);
    ptc.x1[idx] = new_x1;
    ptc.x2[idx] = new_x2;
    ptc.x3[idx] = r3p;

    // printf("c1 %d, c2 %d, x1 %f, x2 %f, v1 %f, v2 %f\n", c1, c2,
    // new_x1,
    //        new_x2, v1, v2);

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

      // size_t j_offset = (j + c2) * fields.J1.p.pitch;
      Scalar djx = 0.0f;
      for (int i = i_0; i <= i_1; i++) {
        Scalar sx0 = interp.interpolate(-old_x1 + i + 1);
        Scalar sx1 = interp.interpolate(-new_x1 + (i + 1 - dc1));

        // j1 is movement in r
        // int offset = j_offset + (i + c1) * sizeof(Scalar);
        Scalar val0 = movement2d(sy0, sy1, sx0, sx1);
        djx += val0;
        // atomicAdd(ptrAddr(fields.J1, offset + sizeof(Scalar)),
        atomicAdd(&fields.J1(i + c1 + 1, j + c2), weight * djx);

        // j2 is movement in theta
        Scalar val1 = movement2d(sx0, sx1, sy0, sy1);
        djy[i - i_0] += val1;
        // atomicAdd(ptrAddr(fields.J2, offset + fields.J2.pitch),
        atomicAdd(&fields.J2(i + c1, j + c2 + 1),
                  weight * djy[i - i_0]);

        // j3 is simply v3 times rho at volume average
        Scalar val2 = center2d(sx0, sx1, sy0, sy1);
        // atomicAdd(ptrAddr(fields.J3, offset),
        atomicAdd(&fields.J3(i + c1, j + c2),
                  -weight * v3 * val2 / mesh_ptrs.dV(i + c1, j + c2));

        // rho is deposited at the final position
        // if ((step + 1) % dev_params.data_interval == 0) {
        Scalar s1 = sx1 * sy1;
        atomicAdd(&fields.Rho[sp](i + c1, j + c2), -weight * s1);
        // }
      }
    }
  }
}

// __global__ void
// convert_j(cudaPitchedPtr j1, cudaPitchedPtr j2,
//           PtcUpdaterDev::fields_data fields) {
//   for (int j = blockIdx.y * blockDim.y + threadIdx.y;
//        j < dev_mesh.dims[1]; j += blockDim.y * gridDim.y) {
//     for (int i = blockIdx.x * blockDim.x + threadIdx.x;
//          i < dev_mesh.dims[0]; i += blockDim.x * gridDim.x) {
//       size_t offset_f = j * fields.J1.pitch + i * sizeof(Scalar);
//       size_t offset_d = j * j1.pitch + i * sizeof(double);
//       (*ptrAddr(fields.J1, offset_f)) =
//           (*(float2 *)((char *)j1.ptr + offset_d)).x;
//       (*ptrAddr(fields.J2, offset_f)) =
//           (*(float2 *)((char *)j2.ptr + offset_d)).x;
//     }
//   }
// }

__global__ void
process_j(PtcUpdaterDev::fields_data fields,
          Grid_LogSph_dev::mesh_ptrs mesh_ptrs, Scalar dt) {
  for (int j = blockIdx.y * blockDim.y + threadIdx.y;
       j < dev_mesh.dims[1]; j += blockDim.y * gridDim.y) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < dev_mesh.dims[0]; i += blockDim.x * gridDim.x) {
      // size_t offset = j * fields.J1.pitch + i * sizeof(Scalar);
      Scalar w = dev_mesh.delta[0] * dev_mesh.delta[1] / dt;
      fields.J1(i, j) *= w / mesh_ptrs.A1_e(i, j);
      fields.J2(i, j) *= w / mesh_ptrs.A2_e(i, j);
      for (int n = 0; n < dev_params.num_species; n++) {
        fields.Rho[n](i, j) /= mesh_ptrs.dV(i, j);
      }
    }
  }
}

__global__ void
inject_ptc(particle_data ptc, size_t num, int inj_per_cell, Scalar p1,
           Scalar p2, Scalar p3, Scalar w,
           // cudaPitchedPtr rho0,
           // cudaPitchedPtr rho1,
           Scalar *surface_e, Scalar *surface_p, curandState *states,
           Scalar omega) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  curandState localState = states[id];
  for (int i = dev_mesh.guard[1] + id;
       // i = dev_mesh.dims[1] - dev_mesh.guard[1] - 3 + id;
       i < dev_mesh.dims[1] - dev_mesh.guard[1];
       i += blockDim.x * gridDim.x) {
    size_t offset = num + i * inj_per_cell * 2;
    Scalar r = exp(dev_mesh.pos(0, dev_mesh.guard[0] + 2, 0.5f));
    // Scalar dens = max(-*ptrAddr(rho0, dev_mesh.guard[0] + 2, i),
    //                   *ptrAddr(rho1, dev_mesh.guard[0] + 2, i));
    Scalar dens = max(surface_e[i - dev_mesh.guard[1]],
                      surface_p[i - dev_mesh.guard[1]]);
    Scalar omega_LT = 0.4f * omega * dev_params.compactness;
    if (dens > 0.4 * square(1.0f / dev_mesh.delta[1])) continue;
    for (int n = 0; n < inj_per_cell; n++) {
      Pos_t x2 = curand_uniform(&localState);
      Scalar theta = dev_mesh.pos(1, i, x2);
      Scalar vphi = (omega - omega_LT) * r * sin(theta);
      // Scalar vphi = omega * r * sin(theta);
      Scalar w_ptc = w * sin(theta) * std::abs(cos(theta));
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
      ptc.weight[offset + n * 2] = w_ptc;
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
      ptc.weight[offset + n * 2 + 1] = w_ptc;
      ptc.flag[offset + n * 2 + 1] = set_ptc_type_flag(
          bit_or(ParticleFlag::primary), ParticleType::ion);
    }
  }
  states[id] = localState;
}

__global__ void
ptc_outflow(particle_data ptc, size_t num) {
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num;
       idx += blockDim.x * gridDim.x) {
    auto c = ptc.cell[idx];
    if (c == MAX_CELL || idx >= num) continue;

    int c1 = dev_mesh.get_c1(c);
    auto flag = ptc.flag[idx];
    if (check_bit(flag, ParticleFlag::ignore_EM)) continue;
    if (c1 > dev_mesh.dims[0] - dev_params.damping_length + 2) {
      flag |= bit_or(ParticleFlag::ignore_EM);
      ptc.flag[idx] = flag;
    }
  }
}

__global__ void
axis_rho_lower(PtcUpdaterDev::fields_data fields,
               Grid_LogSph_dev::mesh_ptrs mesh_ptrs) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < dev_mesh.dims[0]; i += blockDim.x * gridDim.x) {
    // size_t offset_0 =
    //     i * sizeof(Scalar) + dev_mesh.guard[1] * fields.J3.pitch;
    // (*ptrAddr(fields.J3, offset_0 - fields.J3.pitch)) = 0.0f;
    fields.J3(i, dev_mesh.guard[1] - 1) = 0.0f;
    // (*ptrAddr(fields.J2, offset_0 - fields.J2.pitch)) -=
    //     *ptrAddr(fields.J2, offset_0 - 2 * fields.J2.pitch);
    fields.J2(i, dev_mesh.guard[1] - 1) -=
        fields.J2(i, dev_mesh.guard[1] - 2);
  }
}

__global__ void
axis_rho_upper(PtcUpdaterDev::fields_data fields,
               Grid_LogSph_dev::mesh_ptrs mesh_ptrs) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < dev_mesh.dims[0]; i += blockDim.x * gridDim.x) {
    if (i >= dev_mesh.dims[0]) continue;
    // Scalar *row = (Scalar *)((char *)fields.J2.ptr +
    //                          (dev_mesh.dims[1] - dev_mesh.guard[1] -
    //                          1) * fields.J2.pitch);
    // size_t offset_pi =
    //     i * sizeof(Scalar) +
    //     (dev_mesh.dims[1] - dev_mesh.guard[1] - 1) * fields.J3.pitch;
    // (*ptrAddr(fields.J2, offset_pi)) -=
    //     *ptrAddr(fields.J2, offset_pi + fields.J2.pitch);
    fields.J2(i, dev_mesh.dims[1] - dev_mesh.guard[1] - 1) -=
        fields.J2(i, dev_mesh.dims[1] - dev_mesh.guard[1]);

    // (*ptrAddr(fields.J3, offset_pi)) = 0.0f;
    fields.J3(i, dev_mesh.dims[1] - dev_mesh.guard[1] - 1) = 0.0f;
    // printf("%d, %d, %lu, %lu, %lu\n", i,
    //        (dev_mesh.dims[1] - dev_mesh.guard[1] - 1),
    //        fields.J3.pitch, fields.J3.xsize, fields.J3.ysize);
    // (*ptrAddr(fields.J3, offset_pi)) = 0.0f;
    // row[i] = 0.0f;
  }
}

__global__ void
measure_surface_density(particle_data ptc, size_t num,
                        Scalar *surface_e, Scalar *surface_p) {
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num;
       idx += blockDim.x * gridDim.x) {
    auto c = ptc.cell[idx];
    if (c == MAX_CELL || idx >= num) continue;

    // Load particle quantities
    int c1 = dev_mesh.get_c1(c);
    int c2 = dev_mesh.get_c2(c);
    if (c1 >= dev_mesh.guard[0] + 1 && c1 <= dev_mesh.guard[0] + 3) {
      auto flag = ptc.flag[idx];
      int sp = get_ptc_type(flag);
      auto w = ptc.weight[idx];
      if (sp == (int)ParticleType::electron) {
        atomicAdd(surface_e, w / 3.0f);
      } else if (sp == (int)ParticleType::positron) {
        atomicAdd(surface_p, w / 3.0f);
      }
    }
  }
}

__global__ void
annihilate_pairs(particle_data ptc, size_t num,
                 typed_pitchedptr<Scalar> j1,
                 typed_pitchedptr<Scalar> j2,
                 typed_pitchedptr<Scalar> j3) {
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num;
       idx += blockDim.x * gridDim.x) {
    // First do a deposit before annihilation
    auto c = ptc.cell[idx];
    auto flag = ptc.flag[idx];
    // Skip empty particles
    if (c == MAX_CELL || !check_bit(flag, ParticleFlag::annihilate))
      continue;

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

        // size_t j_offset = (j + c2) * j1.pitch;
        Scalar djx = 0.0f;
        for (int i = -1; i <= 0; i++) {
          Scalar sx0 = interp.interpolate(-old_x1 + i + 1);
          Scalar sx1 = interp.interpolate(-new_x1 + i + 1);

          // j1 is movement in r
          // int offset = j_offset + (i + c1) * sizeof(Scalar);
          Scalar val0 = movement2d(sy0, sy1, sx0, sx1);
          djx += val0;
          // atomicAdd(ptrAddr(j1, offset + sizeof(Scalar)), weight *
          // djx);
          atomicAdd(&j1(i + c1 + 1, j + c2), weight * djx);

          // j2 is movement in theta
          Scalar val1 = movement2d(sx0, sx1, sy0, sy1);
          djy[i + 1] += val1;
          // atomicAdd(ptrAddr(j2, offset + j2.pitch),
          atomicAdd(&j2(i + c1, j + c2 + 1), weight * djy[i + 1]);
        }
      }
    }

    // Actually kill the particle
    ptc.cell[idx] = MAX_CELL;
  }
}

__global__ void
flag_annihilation(particle_data data, size_t num,
                  typed_pitchedptr<Scalar> dens,
                  typed_pitchedptr<Scalar> balance) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num;
       i += blockDim.x * gridDim.x) {
    auto c = data.cell[i];
    // Skip empty particles
    if (c == MAX_CELL) continue;

    auto flag = data.flag[i];
    if (get_ptc_type(flag) > 1) continue;  // ignore ions
    auto w = data.weight[i];

    int c1 = dev_mesh.get_c1(c);
    int c2 = dev_mesh.get_c2(c);
    // size_t offset = c1 * sizeof(Scalar) + c2 * dens.pitch;

    Scalar n = atomicAdd(&dens(c1, c2), w);
    Scalar r = std::exp(dev_mesh.pos(0, c1, 0.5f));
    // TODO: implement the proper condition
    if (n >
        0.5 * dev_mesh.inv_delta[0] * dev_mesh.inv_delta[0] / (r * r)) {
      set_bit(flag, ParticleFlag::annihilate);
      atomicAdd(&balance(c1, c2),
                w * (get_ptc_type(flag) == (int)ParticleType::electron
                         ? -1.0f
                         : 1.0f));
    }
  }
  // After this operation, the balance array will contain how much
  // imbalance is there in the annihilated part. We will add this
  // imbalance back in as an extra particle
}

__global__ void
add_extra_particles(particle_data ptc, size_t num,
                    typed_pitchedptr<Scalar> balance) {
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
  Scalar w = balance(n1, n2);
  ptc.weight[num + num_offset] = std::abs(w);
  if (w > 0)
    ptc.flag[num + num_offset] =
        set_ptc_type_flag(0, ParticleType::positron);
  else
    ptc.flag[num + num_offset] =
        set_ptc_type_flag(0, ParticleType::electron);
}

// __global__ void
// filter_current(cudaPitchedPtr j, cudaPitchedPtr j_tmp, cudaPitchedPtr
// A,
//                bool boundary_lower0, bool boundary_upper0,
//                bool boundary_lower1, bool boundary_upper1) {
//   // Load position parameters
//   int t1 = blockIdx.x, t2 = blockIdx.y;
//   int c1 = threadIdx.x, c2 = threadIdx.y;
//   int n1 = dev_mesh.guard[0] + t1 * blockDim.x + c1;
//   int n2 = dev_mesh.guard[1] + t2 * blockDim.y + c2;
//   size_t globalOffset = n2 * j.pitch + n1 * sizeof(Scalar);

//   size_t dr_plus = sizeof(Scalar);
//   if (boundary_upper0 && n1 == dev_mesh.dims[0] - dev_mesh.guard[0] -
//   1)
//     dr_plus = 0;
//   // (n1 < dev_mesh.dims[0] - dev_mesh.guard[0] - 1 ? sizeof(Scalar)
//   //                                                : 0);
//   size_t dr_minus = sizeof(Scalar);
//   if (boundary_lower0 && n1 == dev_mesh.guard[0]) dr_minus = 0;
//   // (n1 > dev_mesh.guard[0] ? sizeof(Scalar) : 0);
//   size_t dt_plus = j.pitch;
//   if (boundary_upper1 && n2 == dev_mesh.dims[1] - dev_mesh.guard[1] -
//   1)
//     dt_plus = 0;
//   // (n2 < dev_mesh.dims[1] - dev_mesh.guard[1] - 1 ? j.pitch : 0);
//   size_t dt_minus = j.pitch;
//   if (boundary_lower1 && n2 == dev_mesh.guard[1]) dt_minus = 0;
//   // (n2 > dev_mesh.guard[1] ? j.pitch : 0);
//   // Do the actual computation here
//   (*ptrAddr(j_tmp, globalOffset)) =
//       0.25f * *ptrAddr(j, globalOffset) * *ptrAddr(A, globalOffset);
//   (*ptrAddr(j_tmp, globalOffset)) +=
//       0.125f * *ptrAddr(j, globalOffset + dr_plus) *
//       *ptrAddr(A, globalOffset + dr_plus);
//   (*ptrAddr(j_tmp, globalOffset)) +=
//       0.125f * *ptrAddr(j, globalOffset - dr_minus) *
//       *ptrAddr(A, globalOffset - dr_minus);
//   (*ptrAddr(j_tmp, globalOffset)) +=
//       0.125f * *ptrAddr(j, globalOffset + dt_plus) *
//       *ptrAddr(A, globalOffset + dt_plus);
//   (*ptrAddr(j_tmp, globalOffset)) +=
//       0.125f * *ptrAddr(j, globalOffset - dt_minus) *
//       *ptrAddr(A, globalOffset - dt_minus);
//   (*ptrAddr(j_tmp, globalOffset)) +=
//       0.0625f * *ptrAddr(j, globalOffset + dr_plus + dt_plus) *
//       *ptrAddr(A, globalOffset + dr_plus + dt_plus);
//   (*ptrAddr(j_tmp, globalOffset)) +=
//       0.0625f * *ptrAddr(j, globalOffset - dr_minus + dt_plus) *
//       *ptrAddr(A, globalOffset - dr_minus + dt_plus);
//   (*ptrAddr(j_tmp, globalOffset)) +=
//       0.0625f * *ptrAddr(j, globalOffset + dr_plus - dt_minus) *
//       *ptrAddr(A, globalOffset + dr_plus - dt_minus);
//   (*ptrAddr(j_tmp, globalOffset)) +=
//       0.0625f * *ptrAddr(j, globalOffset - dr_minus - dt_minus) *
//       *ptrAddr(A, globalOffset - dr_minus - dt_minus);
//   (*ptrAddr(j_tmp, globalOffset)) /= *ptrAddr(A, globalOffset);
// }

}  // namespace Kernels

PtcUpdaterLogSph::PtcUpdaterLogSph(const cu_sim_environment &env)
    : PtcUpdaterDev(env),
      d_rand_states(env.dev_map().size()),
      m_threadsPerBlock(256),
      m_blocksPerGrid(128)
// m_J1(env.local_grid()),
// m_dens(env.local_grid()),
// m_balance(env.local_grid())
{
  // const Grid_LogSph_dev &grid =
  //     dynamic_cast<const Grid_LogSph_dev &>(env.grid());
  // m_mesh_ptrs = grid.get_mesh_ptrs();

  int seed = m_env.params().random_seed;

  for_each_device(env.dev_map(), [this, seed](int n) {
    CudaSafeCall(cudaMalloc(
        &(this->d_rand_states[n]),
        m_threadsPerBlock * m_blocksPerGrid * sizeof(curandState)));
    init_rand_states((curandState *)this->d_rand_states[n], seed + n,
                     this->m_threadsPerBlock, this->m_blocksPerGrid);
  });

  for_each_device(env.dev_map(), [this](int n) {
    m_surface_e.emplace_back(m_env.sub_params(n).N[1],
                             m_env.dev_map()[n]);
    m_surface_p.emplace_back(m_env.sub_params(n).N[1],
                             m_env.dev_map()[n]);
  });
  for_each_device(env.dev_map(), [this](int n) {
    m_tmp_j1.emplace_back(Extent(
        m_env.sub_params(n).N[0] + 2 * m_env.sub_params(n).guard[0],
        m_env.sub_params(n).N[1] + 2 * m_env.sub_params(n).guard[1]));
    m_tmp_j2.emplace_back(Extent(
        m_env.sub_params(n).N[0] + 2 * m_env.sub_params(n).guard[0],
        m_env.sub_params(n).N[1] + 2 * m_env.sub_params(n).guard[1]));
  });
  // m_J1.initialize();
  // m_J2.initialize();
}

PtcUpdaterLogSph::~PtcUpdaterLogSph() {
  for_each_device(m_env.dev_map(), [this](int n) {
    cudaFree((curandState *)this->d_rand_states[n]);
  });
}

void
PtcUpdaterLogSph::update_particles(cu_sim_data &data, double dt,
                                   uint32_t step) {
  timer::stamp("ptc_update");
  initialize_dev_fields(data);

  if (m_env.grid().dim() == 2) {
    for_each_device(data.dev_map, [this, &data](int n) {
      data.J[n].initialize();
      for (int i = 0; i < data.env.params().num_species; i++) {
        data.Rho[i][n].initialize();
      }
    });
    timer::stamp("ptc_push");
    // Skip empty particle array
    for_each_device(data.dev_map, [this, &data, dt, step](int n) {
      if (data.particles[n].number() > 0) {
        Logger::print_info(
            "Updating {} particles in log spherical coordinates",
            data.particles[n].number());
        Kernels::vay_push_2d<<<256, 512>>>(data.particles[n].data(),
                                           data.particles[n].number(),
                                           m_dev_fields[n], dt);
        CudaCheckError();
      }
    });
    for_each_device(data.dev_map, [](int n) {
      CudaSafeCall(cudaDeviceSynchronize());
    });
    timer::show_duration_since_stamp("Pushing particles", "us",
                                     "ptc_push");

    timer::stamp("ptc_deposit");
    for_each_device(data.dev_map, [this, &data, dt, step](int n) {
      if (data.particles[n].number() > 0) {
        const Grid_LogSph_dev *grid =
            dynamic_cast<const Grid_LogSph_dev *>(data.grid[n].get());
        auto mesh_ptrs = grid->get_mesh_ptrs();
        // m_J1.initialize();
        // m_J2.initialize();
        Kernels::deposit_current_2d_log_sph<<<256, 512>>>(
            data.particles[n].data(), data.particles[n].number(),
            m_dev_fields[n], mesh_ptrs, dt, step,
            m_env.is_boundary(n, 2), m_env.is_boundary(n, 3));
        CudaCheckError();
        // Kernels::convert_j<<<dim3(32, 32), dim3(32, 32)>>>(
        //     m_J1.ptr(), m_J2.ptr(), m_dev_fields);
        // CudaCheckError();
      }
    });
    for_each_device(data.dev_map, [](int n) {
      CudaSafeCall(cudaDeviceSynchronize());
    });
    timer::show_duration_since_stamp("Depositing particles", "us",
                                     "ptc_deposit");

    // timer::stamp("comm");
    m_env.send_sub_guard_cells(data.J);
    for (int i = 0; i < data.env.params().num_species; i++) {
      m_env.send_sub_guard_cells(data.Rho[i]);
    }
    for_each_device(data.dev_map, [this, &data, dt, step](int n) {
      const Grid_LogSph_dev *grid =
          dynamic_cast<const Grid_LogSph_dev *>(data.grid[n].get());
      auto mesh_ptrs = grid->get_mesh_ptrs();
      Kernels::process_j<<<dim3(32, 32), dim3(32, 32)>>>(
          m_dev_fields[n], mesh_ptrs, dt);
      CudaCheckError();
    });
    for_each_device(data.dev_map, [](int n) {
      CudaSafeCall(cudaDeviceSynchronize());
    });

    // Logger::print_debug("current smoothing {} times",
    //                     m_env.params().current_smoothing);
    // for (int i = 0; i < m_env.params().current_smoothing; i++) {
    //   m_env.get_sub_guard_cells(data.J);
    //   // Logger::print_debug("current smoothing {}", i);
    //   for_each_device(data.dev_map, [this, &data](int n) {
    //     // Logger::print_debug(
    //     //     "on device {}, tmp_j1 is on {}, tmp_j2 is on {}", n,
    //     //     m_tmp_j1[n].devId(), m_tmp_j2[n].devId());
    //     const Grid_LogSph_dev *grid =
    //         dynamic_cast<const Grid_LogSph_dev
    //         *>(data.grid[n].get());
    //     auto mesh_ptrs = grid->get_mesh_ptrs();
    //     auto &mesh = grid->mesh();
    //     dim3 blockSize(32, 16);
    //     dim3 gridSize(mesh.reduced_dim(0) / 32,
    //                   mesh.reduced_dim(1) / 16);

    //     Kernels::filter_current<<<gridSize, blockSize>>>(
    //         data.J[n].ptr(0), m_tmp_j1[n].data_d(), mesh_ptrs.A1_e,
    //         m_env.is_boundary(n, 0), m_env.is_boundary(n, 1),
    //         m_env.is_boundary(n, 2), m_env.is_boundary(n, 3));
    //     data.J[n].data(0).copy_from(m_tmp_j1[n]);
    //     CudaCheckError();

    //     Kernels::filter_current<<<gridSize, blockSize>>>(
    //         data.J[n].ptr(1), m_tmp_j2[n].data_d(), mesh_ptrs.A2_e,
    //         m_env.is_boundary(n, 0), m_env.is_boundary(n, 1),
    //         m_env.is_boundary(n, 2), m_env.is_boundary(n, 3));
    //     data.J[n].data(1).copy_from(m_tmp_j2[n]);
    //     CudaCheckError();
    //   });
    //   for_each_device(data.dev_map, [](int n) {
    //     CudaSafeCall(cudaDeviceSynchronize());
    //   });
    // }
    // timer::stamp("ph_update");
    for_each_device(data.dev_map, [this, &data, dt, step](int n) {
      // Skip empty particle array
      if (data.photons[n].number() > 0) {
        Logger::print_info(
            "Updating {} photons in log spherical coordinates",
            data.photons[n].number());
        Kernels::move_photons<<<256, 512>>>(
            data.photons[n].data(), data.photons[n].number(), dt,
            m_env.is_boundary(n, 2), m_env.is_boundary(n, 3));
        CudaCheckError();
      }
    });
    for_each_device(data.dev_map, [](int n) {
      CudaSafeCall(cudaDeviceSynchronize());
    });
    // timer::show_duration_since_stamp("Updating photons", "us",
    //                                  "ph_update");
  }
  // timer::show_duration_since_stamp("Sending guard cells", "us",
  // "comm");
  data.send_particles();
  handle_boundary(data);
  timer::show_duration_since_stamp("Ptc update", "us", "ptc_update");
}

void
PtcUpdaterLogSph::handle_boundary(cu_sim_data &data) {
  for_each_device(data.dev_map, [this, &data](int n) {
    data.particles[n].clear_guard_cells();
    data.photons[n].clear_guard_cells();
  });
  for_each_device(data.dev_map,
                  [](int n) { CudaSafeCall(cudaDeviceSynchronize()); });
  for_each_device(data.dev_map, [this, &data](int n) {
    const Grid_LogSph_dev *grid =
        dynamic_cast<const Grid_LogSph_dev *>(data.grid[n].get());
    auto mesh_ptrs = grid->get_mesh_ptrs();

    if (data.env.is_boundary(n, (int)BoundaryPos::lower1)) {
      // CudaSafeCall(cudaSetDevice(n));
      // Logger::print_debug("Processing boundary {} on device {}",
      // (int)BoundaryPos::lower1, n);
      Kernels::axis_rho_lower<<<1, 512>>>(m_dev_fields[n], mesh_ptrs);
      CudaCheckError();
      // cudaDeviceSynchronize();
    }
    if (data.env.is_boundary(n, (int)BoundaryPos::upper1)) {
      // CudaSafeCall(cudaSetDevice(n));
      // Logger::print_debug("Processing boundary {} on device {}",
      //                     (int)BoundaryPos::upper1, n);
      Kernels::axis_rho_upper<<<1, 512>>>(m_dev_fields[n], mesh_ptrs);
      CudaCheckError();
      // cudaDeviceSynchronize();
    }
    if (data.env.is_boundary(n, (int)BoundaryPos::upper0)) {
      Kernels::ptc_outflow<<<256, 512>>>(data.particles[n].data(),
                                         data.particles[n].number());
      CudaCheckError();
    }
  });
  for_each_device(data.dev_map,
                  [](int n) { CudaSafeCall(cudaDeviceSynchronize()); });
}

void
PtcUpdaterLogSph::inject_ptc(cu_sim_data &data, int inj_per_cell,
                             Scalar p1, Scalar p2, Scalar p3, Scalar w,
                             Scalar omega) {
  for_each_device(data.dev_map, [this, &data, &inj_per_cell, &p1, &p2,
                                 &p3, &w, &omega](int n) {
    if (data.env.is_boundary(n, (int)BoundaryPos::lower0)) {
      m_surface_e[n].assign_dev(0.0);
      m_surface_p[n].assign_dev(0.0);
      Kernels::measure_surface_density<<<256, 512>>>(
          data.particles[n].data(), data.particles[n].number(),
          m_surface_e[n].data_d(), m_surface_p[n].data_d());
      CudaCheckError();
      Kernels::inject_ptc<<<m_blocksPerGrid, m_threadsPerBlock>>>(
          data.particles[n].data(), data.particles[n].number(),
          inj_per_cell, p1, p2, p3, w, m_surface_e[n].data_d(),
          m_surface_p[n].data_d(), (curandState *)d_rand_states[n],
          omega);
      CudaCheckError();

      data.particles[n].set_num(
          data.particles[n].number() +
          2 * inj_per_cell * data.E[n].grid().mesh().reduced_dim(1));
    }
  });
}

// void
// PtcUpdaterLogSph::annihilate_extra_pairs(cu_sim_data &data) {
//   m_dens.data().assign_dev(0.0);
//   m_balance.data().assign_dev(0.0);

//   Kernels::flag_annihilation<<<256, 512>>>(
//       data.particles.data(), data.particles.number(), m_dens.ptr(),
//       m_balance.ptr());
//   CudaCheckError();

//   Kernels::annihilate_pairs<<<256, 512>>>(
//       data.particles.data(), data.particles.number(), data.J.ptr(0),
//       data.J.ptr(1), data.J.ptr(2));
//   CudaCheckError();

//   auto &mesh = data.E.grid().mesh();
//   dim3 blockSize(32, 16);
//   dim3 gridSize(mesh.reduced_dim(0) / 32, mesh.reduced_dim(1) / 16);

//   Kernels::add_extra_particles<<<gridSize, blockSize>>>(
//       data.particles.data(), data.particles.number(),
//       m_balance.ptr());
//   CudaCheckError();

//   cudaDeviceSynchronize();
// }

}  // namespace Aperture
