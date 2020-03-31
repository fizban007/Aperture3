#include "algorithms/ptc_updater_logsph.h"
#include "cuda/algorithms/ptc_updater_helper.cu"
#include "cuda/constant_mem.h"
#include "cuda/cudaUtility.h"
#include "cuda/data_ptrs.h"
#include "cuda/grids/grid_log_sph_ptrs.h"
#include "cuda/kernels.h"
#include "cuda/ptr_util.h"
#include "cuda/utils/iterate_devices.h"
#include "sim_data.h"
#include "sim_environment.h"
#include "utils/logger.h"
#include "utils/timer.h"
#include "utils/util_functions.h"

#include "cuda/algorithms/user_push_2d_logsph.cu"

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
__launch_bounds__(256, 4)
vay_push_logsph_2d(data_ptrs data, size_t num, Scalar dt,
                   curandState *states) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  curandState localState = states[tid];
  for (size_t idx = tid; idx < num; idx += blockDim.x * gridDim.x) {
    user_push_2d_logsph<1>(data, idx, dt, localState);
  }
  states[tid] = localState;
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
    if (dev_mesh.pos(1, c2 + dc2, new_x2) < 0.0f) {
      dc2 += 1;
      new_x2 = 1.0f - new_x2;
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

template <int N>
__global__ void
__launch_bounds__(256, 4)
    deposit_and_move_2d_log_sph(data_ptrs data, size_t num,
                                mesh_ptrs_log_sph mesh_ptrs, Scalar dt,
                                uint32_t step, bool axis0, bool axis1) {
  using spline = Spline::spline_t<N>;
  auto &ptc = data.particles;
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num;
       idx += blockDim.x * gridDim.x) {
    auto c = ptc.cell[idx];
    // Skip empty particles
    if (c == MAX_CELL || idx >= num) continue;

    // Load particle quantities
    Interpolator2D<spline> interp;
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

    v1 = v1 / gamma;
    v2 = v2 / gamma;
    v3 = v3 / gamma;
    Scalar v3_gr = v3 - beta_phi(exp_r1, r2);

    // step 1: Compute particle movement and update position
    Scalar x = exp_r1 * std::sin(r2) * std::cos(old_x3);
    Scalar y = exp_r1 * std::sin(r2) * std::sin(old_x3);
    Scalar z = exp_r1 * std::cos(r2);
    // printf("cart position is (%f, %f, %f)\n", x, y, z);

    logsph2cart(v1, v2, v3_gr, r1, r2, old_x3);
    // printf("cart velocity is (%f, %f, %f)\n", v1, v2, v3);
    x += alpha_gr(exp_r1) * v1 * dt;
    y += alpha_gr(exp_r1) * v2 * dt;
    // z += alpha_gr(exp_r1) * (v3 - beta_phi(exp_r1, r2)) * dt;
    z += alpha_gr(exp_r1) * v3_gr * dt;
    // printf("new cart position is (%f, %f, %f)\n", x, y, z);
    Scalar r1p = sqrt(x * x + y * y + z * z);
    Scalar r2p = acos(z / r1p);
    Scalar exp_r1p = r1p;
    r1p = log(r1p);
    Scalar r3p = atan2(y, x);
    // if (x < 0.0f) v1 *= -1.0f;

    // printf("new position is (%f, %f, %f)\n", exp(r1p), r2p, r3p);

    cart2logsph(v1, v2, v3_gr, r1p, r2p, r3p);
    ptc.p1[idx] = v1 * gamma;
    ptc.p2[idx] = v2 * gamma;
    ptc.p3[idx] = (v3_gr + beta_phi(exp_r1p, r2p)) * gamma;

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
    if (dev_mesh.pos(1, c2 + dc2, new_x2) < 0.0f) {
      dc2 += 1;
      new_x2 = 1.0f - new_x2;
      ptc.p2[idx] *= -1.0;
      ptc.p3[idx] *= -1.0;
    }
    if (dev_mesh.pos(1, c2 + dc2, new_x2) >= CONST_PI) {
      dc2 -= 1;
      new_x2 = 1.0f - new_x2;
      ptc.p2[idx] *= -1.0;
      ptc.p3[idx] *= -1.0;
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
    Scalar weight = -dev_charges[sp] * w;

    int j_0 = (dc2 == -1 ? -spline::radius - 1 : -spline::radius);
    int j_1 = (dc2 == 1 ? spline::radius : spline::radius - 1);
    int i_0 = (dc1 == -1 ? -spline::radius - 1 : -spline::radius);
    int i_1 = (dc1 == 1 ? spline::radius : spline::radius - 1);
    Scalar djy[spline::support + 1] = {0.0f};
    for (int j = j_0; j <= j_1; j++) {
      Scalar sy0 = interp.interpolate(-old_x2 + j + 1);
      Scalar sy1 = interp.interpolate(-new_x2 + (j + 1 - dc2));

      size_t j_offset = (j + c2) * data.J1.p.pitch;
      Scalar djx = 0.0f;
      for (int i = i_0; i <= i_1; i++) {
        Scalar sx0 = interp.interpolate(-old_x1 + i + 1);
        Scalar sx1 = interp.interpolate(-new_x1 + (i + 1 - dc1));

        // j1 is movement in r
        int offset = j_offset + (i + c1) * sizeof(Scalar);
        Scalar val0 = movement2d(sy0, sy1, sx0, sx1);
        djx += val0;
        atomicAdd(&data.J1[offset + sizeof(Scalar)], weight * djx);

        // j2 is movement in theta
        Scalar val1 = movement2d(sx0, sx1, sy0, sy1);
        djy[i - i_0] += val1;
        atomicAdd(&data.J2[offset + data.J2.p.pitch],
                  weight * djy[i - i_0]);

        // j3 is simply v3 times rho at volume average
        Scalar val2 = center2d(sx0, sx1, sy0, sy1);
        atomicAdd(&data.J3[offset],
                  // -weight * (v3 - beta_phi(exp_r1, r2)) * val2 /
                  // mesh_ptrs.dV[offset]);
                  -weight * v3_gr * val2 / mesh_ptrs.dV[offset]);

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
process_j(data_ptrs data, mesh_ptrs_log_sph mesh_ptrs, Scalar dt) {
  for (int j = blockIdx.y * blockDim.y + threadIdx.y;
       j < dev_mesh.dims[1]; j += blockDim.y * gridDim.y) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < dev_mesh.dims[0]; i += blockDim.x * gridDim.x) {
      size_t offset = data.J1.compute_offset(i, j);
      Scalar w = dev_mesh.delta[0] * dev_mesh.delta[1] / dt;
      data.J1[offset] *= w / mesh_ptrs.A1_e[offset];
      data.J2[offset] *= w / mesh_ptrs.A2_e[offset];
      for (int n = 0; n < dev_params.num_species; n++) {
        data.Rho[n][offset] /= mesh_ptrs.dV[offset];
      }
    }
  }
}

__global__ void
inject_ptc(data_ptrs data, size_t num, int inj_per_cell, Scalar p1,
           Scalar p2, Scalar p3, Scalar w, Scalar *surface_e,
           Scalar *surface_p, curandState *states, Scalar omega) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  curandState localState = states[id];
  auto &ptc = data.particles;
  // int inject_i = dev_mesh.guard[0] + 3;
  int inject_i = dev_mesh.guard[0] + 2;
  ParticleType p_type =
      (dev_params.inject_ions ? ParticleType::ion
                              : ParticleType::positron);
  for (int i = dev_mesh.guard[1] + 1 + id;
       // i = dev_mesh.dims[1] - dev_mesh.guard[1] - 3 + id;
       i < dev_mesh.dims[1] - dev_mesh.guard[1] - 1;
       i += blockDim.x * gridDim.x) {
    size_t offset = num + i * inj_per_cell * 2;
    Scalar r = exp(dev_mesh.pos(0, inject_i, 0.5f));
    // Scalar dens = max(-*ptrAddr(rho0, dev_mesh.guard[0] + 2, i),
    //                   *ptrAddr(rho1, dev_mesh.guard[0] + 2, i));
    Scalar dens = max(surface_e[i - dev_mesh.guard[1]],
                      surface_p[i - dev_mesh.guard[1]]);
    Scalar omega_LT = 0.4f * omega * dev_params.compactness;
    // if (i == dev_mesh.dims[1] / 2)
    //   printf("dens_e is %f, dens_p is %f, limit is %f\n",
    //          dev_params.q_e * surface_e[i - dev_mesh.guard[1]],
    //          dev_params.q_e * surface_p[i - dev_mesh.guard[1]],
    //          0.4 * square(1.0f / dev_mesh.delta[1]) *
    //              std::sin(dev_mesh.pos(1, i, 0.5f)));
    Scalar sin_theta = std::sin(dev_mesh.pos(1, i, 0.5f));
    if (dev_params.q_e * dens >
        1.5f * square(1.0f / dev_mesh.delta[1]) * sin_theta)
      continue;
    // Scalar Er = data.E1(inject_i, i);
    // Scalar n_inj =
    //     0.2 * std::abs(Er) / (dev_mesh.delta[0] * dev_params.q_e);
    for (int n = 0; n < inj_per_cell; n++) {
    // for (int n = 0; n < n_inj; n++) {
      Pos_t x2 = curand_uniform(&localState);
      Scalar theta = dev_mesh.pos(1, i, x2);
      // Scalar vphi = (omega - omega_LT) * r * sin(theta);
      // Scalar vphi = omega * r * sin(theta);
      Scalar vphi = 0.0f;
      // Scalar w_ptc = w * sin(theta) * std::abs(cos(theta));
      Scalar w_ptc = w * sin(theta);
      // Scalar gamma = 1.0f / std::sqrt(1.0f - vphi * vphi);
      Scalar gamma = std::sqrt(1.0 + p1 * p1 + vphi * vphi);
      float u = curand_uniform(&localState);
      ptc.x1[offset + n * 2] = 0.5f;
      ptc.x2[offset + n * 2] = x2;
      ptc.x3[offset + n * 2] = 0.0f;
      ptc.p1[offset + n * 2] = p1 * 2.0f * std::abs(cos(theta));
      ptc.p2[offset + n * 2] = p1 * sin(theta) * sgn(cos(theta));
      // ptc.p1[offset + n * 2] = p1;
      // ptc.p2[offset + n * 2] = p2;
      ptc.p3[offset + n * 2] = vphi;
      ptc.E[offset + n * 2] = gamma;
      // sqrt(1.0f + p1 * p1 + p2 * p2 + vphi * vphi);
      // printf("inject E is %f\n", ptc.E[offset + n * 2]);
      // ptc.p3[offset + n * 2] = p3;
      ptc.cell[offset + n * 2] = dev_mesh.get_idx(inject_i, i);
      ptc.weight[offset + n * 2] = w_ptc;
      ptc.flag[offset + n * 2] = set_ptc_type_flag(
          (u < dev_params.track_percent
               ? bit_or(ParticleFlag::primary, ParticleFlag::tracked)
               : bit_or(ParticleFlag::primary)),
          ParticleType::electron);

      ptc.x1[offset + n * 2 + 1] = 0.5f;
      ptc.x2[offset + n * 2 + 1] = x2;
      ptc.x3[offset + n * 2 + 1] = 0.0f;
      ptc.p1[offset + n * 2 + 1] = p1 * 2.0f * std::abs(cos(theta));
      ptc.p2[offset + n * 2 + 1] = p1 * sin(theta) * sgn(cos(theta));
      // ptc.p1[offset + n * 2 + 1] = p1;
      // ptc.p2[offset + n * 2 + 1] = p2;
      ptc.p3[offset + n * 2 + 1] = vphi;
      ptc.E[offset + n * 2 + 1] = gamma;
      // sqrt(1.0f + p1 * p1 + p2 * p2 + vphi * vphi);
      // printf("inject E is %f\n", ptc.E[offset + n * 2 + 1]);
      // ptc.p3[offset + n * 2 + 1] = p3;
      ptc.cell[offset + n * 2 + 1] = dev_mesh.get_idx(inject_i, i);
      ptc.weight[offset + n * 2 + 1] = w_ptc;
      ptc.flag[offset + n * 2 + 1] = set_ptc_type_flag(
          (u < dev_params.track_percent
               ? bit_or(ParticleFlag::primary, ParticleFlag::tracked)
               : bit_or(ParticleFlag::primary)),
          p_type);
      if (u < dev_params.track_percent) {
        ptc.id[offset + n * 2] = dev_rank + atomicAdd(&dev_ptc_id, 1);
        ptc.id[offset + n * 2 + 1] = dev_rank + atomicAdd(&dev_ptc_id, 1);
      }
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
axis_rho_lower(data_ptrs data, mesh_ptrs_log_sph mesh_ptrs) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < dev_mesh.dims[0]; i += blockDim.x * gridDim.x) {
    int j_0 = dev_mesh.guard[1];
    data.J3(i, j_0 - 1) = 0.0f;
    data.J3(i, j_0) = 0.0f;
    // fields.J2(i, j_0) -= fields.J2(i, j_0 - 1);
    // fields.J2(i, j_0 - 1) = 0.0;
  }
}

__global__ void
axis_rho_upper(data_ptrs data, mesh_ptrs_log_sph mesh_ptrs) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < dev_mesh.dims[0]; i += blockDim.x * gridDim.x) {
    if (i >= dev_mesh.dims[0]) continue;
    int j_last = dev_mesh.dims[1] - dev_mesh.guard[1];
    // fields.J2(i, j_last - 1) -= fields.J2(i, j_last);
    // fields.J2(i, j_last) = 0.0;

    data.J3(i, j_last) = 0.0f;
    data.J3(i, j_last - 1) = 0.0f;
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
    // Sum over 3 cells, hense the w / 3.0f in the atomicAdd
    int sum_cells = 3;
    // int inject_cell = dev_mesh.guard[0] + 3;
    int inject_cell = dev_mesh.guard[0] + 2;
    if (c1 >= inject_cell - 1 && c1 <= inject_cell - 1 + sum_cells) {
      auto flag = ptc.flag[idx];
      int sp = get_ptc_type(flag);
      auto w = ptc.weight[idx];
      if (sp == (int)ParticleType::electron) {
        atomicAdd(&surface_e[max(c2 - dev_mesh.guard[1] - 2, 0)],
                  1.0f * w / float(sum_cells) / 16.0f);
        atomicAdd(&surface_e[max(c2 - dev_mesh.guard[1] - 1, 0)],
                  4.0f * w / float(sum_cells) / 16.0f);
        atomicAdd(&surface_e[c2 - dev_mesh.guard[1]],
                  6.0f * w / float(sum_cells) / 16.0f);
        atomicAdd(&surface_e[min(
                      c2 - dev_mesh.guard[1] + 1,
                      dev_mesh.dims[1] - 2 * dev_mesh.guard[1] - 1)],
                  4.0f * w / float(sum_cells) / 16.0f);
        atomicAdd(&surface_e[min(
                      c2 - dev_mesh.guard[1] + 2,
                      dev_mesh.dims[1] - 2 * dev_mesh.guard[1] - 1)],
                  1.0f * w / float(sum_cells) / 16.0f);
      } else if (sp == (int)ParticleType::ion) {
        // } else if (sp == (int)ParticleType::positron) {
        atomicAdd(&surface_p[max(c2 - dev_mesh.guard[1] - 2, 0)],
                  1.0f * w / float(sum_cells) / 16.0f);
        atomicAdd(&surface_p[max(c2 - dev_mesh.guard[1] - 1, 0)],
                  4.0f * w / float(sum_cells) / 16.0f);
        atomicAdd(&surface_p[c2 - dev_mesh.guard[1]],
                  6.0f * w / float(sum_cells) / 16.0f);
        atomicAdd(&surface_p[min(
                      c2 - dev_mesh.guard[1] + 1,
                      dev_mesh.dims[1] - 2 * dev_mesh.guard[1] - 1)],
                  4.0f * w / float(sum_cells) / 16.0f);
        atomicAdd(&surface_p[min(
                      c2 - dev_mesh.guard[1] + 2,
                      dev_mesh.dims[1] - 2 * dev_mesh.guard[1] - 1)],
                  1.0f * w / float(sum_cells) / 16.0f);
      }
    }
  }
}

__global__ void
annihilate_pairs(particle_data ptc, size_t num, pitchptr<Scalar> j1,
                 pitchptr<Scalar> j2, pitchptr<Scalar> j3) {
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
          Scalar val0 = movement2d(sy0, sy1, sx0, sx1);
          djx += val0;
          atomicAdd(&j1(i + c1 + 1, j + c2), weight * djx);

          // j2 is movement in theta
          Scalar val1 = movement2d(sx0, sx1, sy0, sy1);
          djy[i + 1] += val1;
          atomicAdd(&j2(i + c1, j + c2 + 1), weight * djy[i + 1]);
        }
      }
    }

    // Actually kill the particle
    ptc.cell[idx] = MAX_CELL;
  }
}

__global__ void
flag_annihilation(particle_data data, size_t num, pitchptr<Scalar> dens,
                  pitchptr<Scalar> balance) {
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
                    pitchptr<Scalar> balance) {
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

__global__ void
filter_current_logsph(pitchptr<Scalar> j, pitchptr<Scalar> j_tmp,
                      pitchptr<Scalar> A, bool boundary_lower0,
                      bool boundary_upper0, bool boundary_lower1,
                      bool boundary_upper1) {
  // Load position parameters
  int t1 = blockIdx.x, t2 = blockIdx.y;
  int c1 = threadIdx.x, c2 = threadIdx.y;
  int n1 = dev_mesh.guard[0] + t1 * blockDim.x + c1;
  int n2 = dev_mesh.guard[1] + t2 * blockDim.y + c2;
  // size_t globalOffset = n2 * j.p.pitch + n1 * sizeof(Scalar);
  size_t globalOffset = j.compute_offset(n1, n2);

  size_t dr_plus = sizeof(Scalar);
  if (boundary_upper0 && n1 == dev_mesh.dims[0] - dev_mesh.guard[0] - 1)
    dr_plus = 0;
  // (n1 < dev_mesh.dims[0] - dev_mesh.guard[0] - 1 ? sizeof(Scalar)
  //                                                : 0);
  size_t dr_minus = sizeof(Scalar);
  if (boundary_lower0 && n1 == dev_mesh.guard[0]) dr_minus = 0;
  // (n1 > dev_mesh.guard[0] ? sizeof(Scalar) : 0);
  size_t dt_plus = j.p.pitch;
  if (boundary_upper1 && n2 == dev_mesh.dims[1] - dev_mesh.guard[1] - 1)
    dt_plus = 0;
  // (n2 < dev_mesh.dims[1] - dev_mesh.guard[1] - 1 ? j.pitch : 0);
  size_t dt_minus = j.p.pitch;
  if (boundary_lower1 && n2 == dev_mesh.guard[1]) dt_minus = 0;
  // (n2 > dev_mesh.guard[1] ? j.pitch : 0);
  // Do the actual computation here
  j_tmp[globalOffset] = 0.25f * j[globalOffset] * A[globalOffset];
  j_tmp[globalOffset] +=
      0.125f * j[globalOffset + dr_plus] * A[globalOffset + dr_plus];
  j_tmp[globalOffset] +=
      0.125f * j[globalOffset - dr_minus] * A[globalOffset - dr_minus];
  j_tmp[globalOffset] +=
      0.125f * j[globalOffset + dt_plus] * A[globalOffset + dt_plus];
  j_tmp[globalOffset] +=
      0.125f * j[globalOffset - dt_minus] * A[globalOffset - dt_minus];
  j_tmp[globalOffset] += 0.0625f * j[globalOffset + dr_plus + dt_plus] *
                         A[globalOffset + dr_plus + dt_plus];
  j_tmp[globalOffset] += 0.0625f *
                         j[globalOffset - dr_minus + dt_plus] *
                         A[globalOffset - dr_minus + dt_plus];
  j_tmp[globalOffset] += 0.0625f *
                         j[globalOffset + dr_plus - dt_minus] *
                         A[globalOffset + dr_plus - dt_minus];
  j_tmp[globalOffset] += 0.0625f *
                         j[globalOffset - dr_minus - dt_minus] *
                         A[globalOffset - dr_minus - dt_minus];
  j_tmp[globalOffset] /= A[globalOffset];
}

}  // namespace Kernels

ptc_updater_logsph::ptc_updater_logsph(sim_environment &env)
    : m_env(env),
      m_surface_e(env.params().N[1]),
      m_surface_p(env.params().N[1]),
      m_surface_tmp(env.params().N[1]) {
  m_tmp_j1 = multi_array<Scalar>(env.local_grid().extent());
  m_tmp_j2 = multi_array<Scalar>(env.local_grid().extent());
}

ptc_updater_logsph::~ptc_updater_logsph() {}

void
ptc_updater_logsph::update_particles(sim_data &data, double dt,
                                     uint32_t step) {
  timer::stamp("ptc_update");
  auto data_p = get_data_ptrs(data);

  if (m_env.grid().dim() == 2) {
    data.J.initialize();
    for (int i = 0; i < data.env.params().num_species; i++) {
      data.Rho[i].initialize();
    }
    Grid_LogSph *grid =
        dynamic_cast<Grid_LogSph *>(&m_env.local_grid());
    auto mesh_ptrs = get_mesh_ptrs(*grid);
    timer::stamp("ptc_push");
    // Skip empty particle array
    if (data.particles.number() > 0) {
      Logger::print_info(
          "Updating {} particles in log spherical coordinates",
          data.particles.number());
      Kernels::vay_push_logsph_2d<<<512, 256>>>(
          data_p, data.particles.number(), dt,
          (curandState *)data.d_rand_states);
      CudaCheckError();
    }
    CudaSafeCall(cudaDeviceSynchronize());
    timer::show_duration_since_stamp("Pushing particles", "ms",
                                     "ptc_push");

    timer::stamp("ptc_deposit");

    if (data.particles.number() > 0) {
      // m_J1.initialize();
      // m_J2.initialize();
      Kernels::deposit_and_move_2d_log_sph<1><<<512, 256>>>(
          data_p, data.particles.number(), mesh_ptrs, dt, step,
          m_env.is_boundary(2), m_env.is_boundary(3));
      CudaSafeCall(cudaDeviceSynchronize());
      CudaCheckError();
      // Kernels::convert_j<<<dim3(32, 32), dim3(32, 32)>>>(
      //     m_J1.ptr(), m_J2.ptr(), m_dev_fields);
      // CudaCheckError();
    }
    timer::show_duration_since_stamp("Depositing particles", "ms",
                                     "ptc_deposit");

    Kernels::process_j<<<dim3(32, 32), dim3(32, 32)>>>(data_p,
                                                       mesh_ptrs, dt);
    CudaCheckError();
    CudaSafeCall(cudaDeviceSynchronize());

    // timer::stamp("comm");
    // m_env.send_sub_guard_cells(data.J);
    m_env.send_add_guard_cells(data.J);
    m_env.send_guard_cells(data.J);
    for (int i = 0; i < data.env.params().num_species; i++) {
      m_env.send_add_guard_cells(data.Rho[i]);
      m_env.send_guard_cells(data.Rho[i]);
    }

    Logger::print_debug("current smoothing {} times",
                        m_env.params().current_smoothing);
    for (int i = 0; i < m_env.params().current_smoothing; i++) {
      auto &mesh = grid->mesh();
      dim3 blockSize(32, 16);
      dim3 gridSize(mesh.reduced_dim(0) / 32, mesh.reduced_dim(1) / 16);

      Kernels::filter_current_logsph<<<gridSize, blockSize>>>(
          get_pitchptr(data.J.data(0)), get_pitchptr(m_tmp_j1),
          mesh_ptrs.A1_e, m_env.is_boundary(0), m_env.is_boundary(1),
          m_env.is_boundary(2), m_env.is_boundary(3));
      data.J.data(0).copy_from(m_tmp_j1);
      CudaCheckError();

      Kernels::filter_current_logsph<<<gridSize, blockSize>>>(
          get_pitchptr(data.J.data(1)), get_pitchptr(m_tmp_j1),
          mesh_ptrs.A2_e, m_env.is_boundary(0), m_env.is_boundary(1),
          m_env.is_boundary(2), m_env.is_boundary(3));
      data.J.data(1).copy_from(m_tmp_j1);
      CudaCheckError();

      Kernels::filter_current_logsph<<<gridSize, blockSize>>>(
          get_pitchptr(data.J.data(2)), get_pitchptr(m_tmp_j1),
          mesh_ptrs.A3_e, m_env.is_boundary(0), m_env.is_boundary(1),
          m_env.is_boundary(2), m_env.is_boundary(3));
      data.J.data(2).copy_from(m_tmp_j1);
      CudaCheckError();

      if ((step + 1) % data.env.params().data_interval == 0) {
        for (int i = 0; i < data.env.params().num_species; i++) {
          Kernels::filter_current_logsph<<<gridSize, blockSize>>>(
              get_pitchptr(data.Rho[i].data()), get_pitchptr(m_tmp_j1),
              mesh_ptrs.dV, m_env.is_boundary(0), m_env.is_boundary(1),
              m_env.is_boundary(2), m_env.is_boundary(3));
          data.Rho[i].data().copy_from(m_tmp_j1);
          CudaCheckError();
        }
      }
      CudaSafeCall(cudaDeviceSynchronize());
      m_env.send_guard_cells(data.J);
      if ((step + 1) % data.env.params().data_interval == 0) {
        for (int i = 0; i < data.env.params().num_species; i++) {
          m_env.send_guard_cells(data.Rho[i]);
        }
      }
    }
    // timer::stamp("ph_update");
    // Skip empty particle array
    // timer::show_duration_since_stamp("Updating photons", "us",
    //                                  "ph_update");
  }
  m_env.send_particles(data.particles);
  if (data.photons.number() > 0) {
    Logger::print_info(
        "Updating {} photons in log spherical coordinates",
        data.photons.number());
    Kernels::move_photons<<<256, 512>>>(
        data.photons.data(), data.photons.number(), dt,
        m_env.is_boundary(2), m_env.is_boundary(3));
    CudaCheckError();
  }
  m_env.send_particles(data.photons);
  CudaSafeCall(cudaDeviceSynchronize());
  // timer::show_duration_since_stamp("Sending guard cells", "us",
  // "comm");
  // data.send_particles();

  apply_boundary(data, dt, step);
  timer::show_duration_since_stamp("Ptc update", "ms", "ptc_update");
}

void
ptc_updater_logsph::apply_boundary(sim_data &data, double dt,
                                   uint32_t step) {
  auto data_p = get_data_ptrs(data);
  if (data.env.is_boundary((int)BoundaryPos::lower0)) {
  }
  data.particles.clear_guard_cells(m_env.local_grid());
  data.photons.clear_guard_cells(m_env.local_grid());
  CudaSafeCall(cudaDeviceSynchronize());
  Grid_LogSph *grid = dynamic_cast<Grid_LogSph *>(&m_env.local_grid());
  auto mesh_ptrs = get_mesh_ptrs(*grid);

  if (data.env.is_boundary((int)BoundaryPos::lower1)) {
    // CudaSafeCall(cudaSetDevice(n));
    // Logger::print_debug("Processing boundary {} on device {}",
    // (int)BoundaryPos::lower1, n);
    Kernels::axis_rho_lower<<<1, 512>>>(data_p, mesh_ptrs);
    CudaCheckError();
    // cudaDeviceSynchronize();
  }
  if (data.env.is_boundary((int)BoundaryPos::upper1)) {
    // CudaSafeCall(cudaSetDevice(n));
    // Logger::print_debug("Processing boundary {} on device {}",
    //                     (int)BoundaryPos::upper1, n);
    Kernels::axis_rho_upper<<<1, 512>>>(data_p, mesh_ptrs);
    CudaCheckError();
    // cudaDeviceSynchronize();
  }
  if (data.env.is_boundary((int)BoundaryPos::upper0)) {
    Kernels::ptc_outflow<<<256, 512>>>(data.particles.data(),
                                       data.particles.number());
    CudaCheckError();
  }
  CudaSafeCall(cudaDeviceSynchronize());
}

void
ptc_updater_logsph::inject_ptc(sim_data &data, int inj_per_cell,
                               Scalar p1, Scalar p2, Scalar p3,
                               Scalar w, Scalar omega) {
  if (data.env.is_boundary((int)BoundaryPos::lower0)) {
    m_surface_e.assign_dev(0.0);
    m_surface_p.assign_dev(0.0);
    m_surface_tmp.assign_dev(0.0);
    Kernels::measure_surface_density<<<256, 512>>>(
        data.particles.data(), data.particles.number(),
        m_surface_e.dev_ptr(), m_surface_p.dev_ptr());
    CudaCheckError();
    Kernels::inject_ptc<<<128, 256>>>(
        get_data_ptrs(data), data.particles.number(), inj_per_cell, p1,
        p2, p3, w, m_surface_e.dev_ptr(), m_surface_p.dev_ptr(),
        (curandState *)data.d_rand_states, omega);
    CudaCheckError();

    data.particles.set_num(data.particles.number() +
                           2 * inj_per_cell *
                               data.E.grid().mesh().reduced_dim(1));
  }
}

// void
// ptc_updater_logsph::annihilate_extra_pairs(sim_data &data) {
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
