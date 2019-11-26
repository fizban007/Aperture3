#include "algorithms/ptc_updater_cart.h"
#include "cuda/algorithms/ptc_updater_helper.cuh"
#include "cuda/constant_mem.h"
#include "cuda/cudaUtility.h"
#include "cuda/data_ptrs.h"
#include "cuda/kernels.h"
// #include "cuda/ptr_util.h"
#include "sim_data.h"
#include "sim_environment.h"
#include "utils/logger.h"
#include "utils/timer.h"
#include "utils/util_functions.h"

#include "cuda/algorithms/user_push_2d_cart.cuh"

namespace Aperture {

namespace Kernels {

typedef Spline::cloud_in_cell spline_t;

__global__ void
vay_push_2d_cart(data_ptrs data, size_t num, Scalar dt,
                 curandState *states) {
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  curandState localState = states[idx];
  for (; idx < num; idx += blockDim.x * gridDim.x) {
    user_push_2d_cart(data, idx, dt, localState);
  }
  states[idx] = localState;
}

__global__ void
move_photons_cart(photon_data photons, size_t num, Scalar dt,
                  bool axis0, bool axis1) {
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
    // Scalar x = dev_mesh.pos(0, c1, old_x1);
    // Scalar y = dev_mesh.pos(1, c2, old_x2);
    Scalar z = old_x3;

    z += v3 * dt;

    Pos_t new_x1 = old_x1 + (v1 * dt) / dev_mesh.delta[0];
    Pos_t new_x2 = old_x2 + (v2 * dt) / dev_mesh.delta[1];
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
    photons.x3[idx] = z;
    photons.path_left[idx] -= dt;
  }
}

__global__ void
__launch_bounds__(512, 4)
    deposit_current_2d_cart(data_ptrs data, size_t num, Scalar dt,
                            uint32_t step, bool axis0, bool axis1) {
  auto &ptc = data.particles;
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

    // Scalar old_pos3 =
    Pos_t new_x1 = old_x1 + v1 * dt / dev_mesh.delta[0];
    Pos_t new_x2 = old_x2 + v2 * dt / dev_mesh.delta[1];
    Pos_t new_x3 = old_x3 + v3 * dt;
    // printf("new_x1 is %f, new_x2 is %f, old_x1 is %f,
    // old_x2 is %f\n", new_x1, new_x2, old_x1, old_x2);
    int dc1 = floor(new_x1);
    int dc2 = floor(new_x2);
#ifndef NDEBUG
    if (dc1 > 1 || dc1 < -1 || dc2 > 1 || dc2 < -1)
      printf("----------------- Error: moved more than 1 cell!");
#endif
    new_x1 -= (Pos_t)dc1;
    new_x2 -= (Pos_t)dc2;

    ptc.cell[idx] = dev_mesh.get_idx(c1 + dc1, c2 + dc2);
    printf("new_x1 is %f, new_x2 is %f, dc2 = %d\n", new_x1, new_x2,
           dc2);
    ptc.x1[idx] = new_x1;
    ptc.x2[idx] = new_x2;
    ptc.x3[idx] = new_x3;

    // printf("c1 %d, c2 %d, x1 %f, x2 %f, v1 %f, v2 %f\n", c1, c2,
    // new_x1,
    //        new_x2, v1, v2);

    // step 2: Deposit current
    if (check_bit(flag, ParticleFlag::ignore_current)) continue;
    Scalar weight = -dev_charges[sp] * w;

    int j_0 = (dc2 == -1 ? -2 : -1);
    int j_1 = (dc2 == 1 ? 1 : 0);
    int i_0 = (dc1 == -1 ? -2 : -1);
    int i_1 = (dc1 == 1 ? 1 : 0);
    Scalar djy[3] = {0.0f};
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
                  -weight * v3 * val2);

        // rho is deposited at the final position
        if ((step + 1) % dev_params.data_interval == 0) {
          Scalar s1 = sx1 * sy1;
          atomicAdd(&data.Rho[sp][offset], -weight * s1);
        }
      }
    }
  }
}

// __global__ void
// process_j(data_ptrs data, mesh_ptrs_log_sph mesh_ptrs, Scalar dt) {
//   for (int j = blockIdx.y * blockDim.y + threadIdx.y;
//        j < dev_mesh.dims[1]; j += blockDim.y * gridDim.y) {
//     for (int i = blockIdx.x * blockDim.x + threadIdx.x;
//          i < dev_mesh.dims[0]; i += blockDim.x * gridDim.x) {
//       size_t offset = data.J1.compute_offset(i, j);
//       Scalar w = dev_mesh.delta[0] * dev_mesh.delta[1] / dt;
//       data.J1[offset] *= w / mesh_ptrs.A1_e[offset];
//       data.J2[offset] *= w / mesh_ptrs.A2_e[offset];
//       for (int n = 0; n < dev_params.num_species; n++) {
//         data.Rho[n][offset] /= mesh_ptrs.dV[offset];
//       }
//     }
//   }
// }

// __global__ void
// inject_ptc(particle_data ptc, size_t num, int inj_per_cell, Scalar
// p1,
//            Scalar p2, Scalar p3, Scalar w, Scalar *surface_e,
//            Scalar *surface_p, curandState *states, Scalar omega) {
//   int id = threadIdx.x + blockIdx.x * blockDim.x;
//   curandState localState = states[id];
//   // int inject_i = dev_mesh.guard[0] + 3;
//   int inject_i = dev_mesh.guard[0] + 2;
//   ParticleType p_type =
//       (dev_params.inject_ions ? ParticleType::ion
//                               : ParticleType::positron);
//   for (int i = dev_mesh.guard[1] + 1 + id;
//        // i = dev_mesh.dims[1] - dev_mesh.guard[1] - 3 + id;
//        i < dev_mesh.dims[1] - dev_mesh.guard[1] - 1;
//        i += blockDim.x * gridDim.x) {
//     size_t offset = num + i * inj_per_cell * 2;
//     Scalar r = exp(dev_mesh.pos(0, inject_i, 0.5f));
//     // Scalar dens = max(-*ptrAddr(rho0, dev_mesh.guard[0] + 2, i),
//     //                   *ptrAddr(rho1, dev_mesh.guard[0] + 2, i));
//     Scalar dens = max(surface_e[i - dev_mesh.guard[1]],
//                       surface_p[i - dev_mesh.guard[1]]);
//     Scalar omega_LT = 0.4f * omega * dev_params.compactness;
//     // if (i == dev_mesh.dims[1] / 2)
//     //   printf("dens_e is %f, dens_p is %f, limit is %f\n",
//     //          dev_params.q_e * surface_e[i - dev_mesh.guard[1]],
//     //          dev_params.q_e * surface_p[i - dev_mesh.guard[1]],
//     //          0.4 * square(1.0f / dev_mesh.delta[1]) *
//     //              std::sin(dev_mesh.pos(1, i, 0.5f)));
//     //if (dev_params.q_e * dens > 0.5f *
//     //                                square(1.0f /
//     dev_mesh.delta[1]) *
//     //                                std::sin(dev_mesh.pos(1, i,
//     0.5f)))
//     //  continue;
//     for (int n = 0; n < inj_per_cell; n++) {
//       Pos_t x2 = curand_uniform(&localState);
//       Scalar theta = dev_mesh.pos(1, i, x2);
//       // Scalar vphi = (omega - omega_LT) * r * sin(theta);
//       // Scalar vphi = omega * r * sin(theta);
//       Scalar vphi = 0.0f;
//       // Scalar w_ptc = w * sin(theta) * std::abs(cos(theta));
//       Scalar w_ptc = w * sin(theta);
//       // Scalar gamma = 1.0f / std::sqrt(1.0f - vphi * vphi);
//       Scalar gamma = std::sqrt(1.0 + p1 * p1 + vphi * vphi);
//       float u = curand_uniform(&localState);
//       ptc.x1[offset + n * 2] = 0.5f;
//       ptc.x2[offset + n * 2] = x2;
//       ptc.x3[offset + n * 2] = 0.0f;
//       ptc.p1[offset + n * 2] = p1 * 2.0f * std::abs(cos(theta));
//       ptc.p2[offset + n * 2] = p1 * sin(theta) * sgn(cos(theta));
//       ptc.p3[offset + n * 2] = vphi;
//       ptc.E[offset + n * 2] = gamma;
//       // sqrt(1.0f + p1 * p1 + p2 * p2 + vphi * vphi);
//       // printf("inject E is %f\n", ptc.E[offset + n * 2]);
//       // ptc.p3[offset + n * 2] = p3;
//       ptc.cell[offset + n * 2] = dev_mesh.get_idx(inject_i, i);
//       ptc.weight[offset + n * 2] = w_ptc;
//       ptc.flag[offset + n * 2] = set_ptc_type_flag(
//           (u < dev_params.track_percent
//                ? bit_or(ParticleFlag::primary, ParticleFlag::tracked)
//                : bit_or(ParticleFlag::primary)),
//           ParticleType::electron);

//       ptc.x1[offset + n * 2 + 1] = 0.5f;
//       ptc.x2[offset + n * 2 + 1] = x2;
//       ptc.x3[offset + n * 2 + 1] = 0.0f;
//       ptc.p1[offset + n * 2 + 1] = p1 * 2.0f * std::abs(cos(theta));
//       ptc.p2[offset + n * 2 + 1] = p1 * sin(theta) * sgn(cos(theta));
//       ptc.p3[offset + n * 2 + 1] = vphi;
//       ptc.E[offset + n * 2 + 1] = gamma;
//       // sqrt(1.0f + p1 * p1 + p2 * p2 + vphi * vphi);
//       // printf("inject E is %f\n", ptc.E[offset + n * 2 + 1]);
//       // ptc.p3[offset + n * 2 + 1] = p3;
//       ptc.cell[offset + n * 2 + 1] = dev_mesh.get_idx(inject_i, i);
//       ptc.weight[offset + n * 2 + 1] = w_ptc;
//       ptc.flag[offset + n * 2 + 1] = set_ptc_type_flag(
//           (u < dev_params.track_percent
//                ? bit_or(ParticleFlag::primary, ParticleFlag::tracked)
//                : bit_or(ParticleFlag::primary)),
//           p_type);
//       if (u < dev_params.track_percent) {
//         ptc.id[offset + n * 2] = atomicAdd(&dev_ptc_id, 1);
//         ptc.id[offset + n * 2 + 1] = atomicAdd(&dev_ptc_id, 1);
//       }
//     }
//   }
//   states[id] = localState;
// }

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
filter_current_cart(pitchptr<Scalar> j, pitchptr<Scalar> j_tmp,
                    bool boundary_lower0, bool boundary_upper0,
                    bool boundary_lower1, bool boundary_upper1) {
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
  j_tmp[globalOffset] = 0.25f * j[globalOffset];
  j_tmp[globalOffset] += 0.125f * j[globalOffset + dr_plus];
  j_tmp[globalOffset] += 0.125f * j[globalOffset - dr_minus];
  j_tmp[globalOffset] += 0.125f * j[globalOffset + dt_plus];
  j_tmp[globalOffset] += 0.125f * j[globalOffset - dt_minus];
  j_tmp[globalOffset] += 0.0625f * j[globalOffset + dr_plus + dt_plus];
  j_tmp[globalOffset] += 0.0625f * j[globalOffset - dr_minus + dt_plus];
  j_tmp[globalOffset] += 0.0625f * j[globalOffset + dr_plus - dt_minus];
  j_tmp[globalOffset] +=
      0.0625f * j[globalOffset - dr_minus - dt_minus];
  // j_tmp[globalOffset] /= A[globalOffset];
}

}  // namespace Kernels

ptc_updater_cart::ptc_updater_cart(sim_environment &env) : m_env(env) {
  m_tmp_j1 = multi_array<Scalar>(env.local_grid().extent());
  m_tmp_j2 = multi_array<Scalar>(env.local_grid().extent());
}

ptc_updater_cart::~ptc_updater_cart() {}

void
ptc_updater_cart::update_particles(sim_data &data, double dt,
                                   uint32_t step) {
  timer::stamp("ptc_update");
  auto data_p = get_data_ptrs(data);
  auto &grid = m_env.grid();

  if (grid.dim() == 2) {
    data.J.initialize();
    for (int i = 0; i < data.env.params().num_species; i++) {
      data.Rho[i].initialize();
    }
    timer::stamp("ptc_push");
    // Skip empty particle array
    if (data.particles.number() > 0) {
      Logger::print_info(
          "Updating {} particles in log spherical coordinates",
          data.particles.number());
      Kernels::vay_push_2d_cart<<<256, 512>>>(
          data_p, data.particles.number(), dt,
          (curandState *)data.d_rand_states);
      CudaCheckError();
    }
    CudaSafeCall(cudaDeviceSynchronize());
    timer::show_duration_since_stamp("Pushing particles", "us",
                                     "ptc_push");

    timer::stamp("ptc_deposit");

    if (data.particles.number() > 0) {
      // m_J1.initialize();
      // m_J2.initialize();
      Kernels::deposit_current_2d_cart<<<256, 512>>>(
          data_p, data.particles.number(), dt, step,
          m_env.is_boundary(2), m_env.is_boundary(3));
      CudaCheckError();
      // Kernels::convert_j<<<dim3(32, 32), dim3(32, 32)>>>(
      //     m_J1.ptr(), m_J2.ptr(), m_dev_fields);
      // CudaCheckError();
    }
    CudaSafeCall(cudaDeviceSynchronize());
    timer::show_duration_since_stamp("Depositing particles", "us",
                                     "ptc_deposit");

    // timer::stamp("comm");
    // m_env.send_sub_guard_cells(data.J);
    // for (int i = 0; i < data.env.params().num_species; i++) {
    //   m_env.send_sub_guard_cells(data.Rho[i]);
    // }

    Logger::print_debug("current smoothing {} times",
                        m_env.params().current_smoothing);
    for (int i = 0; i < m_env.params().current_smoothing; i++) {
      // m_env.get_sub_guard_cells(data.J);
      // if ((step + 1) % data.env.params().data_interval == 0) {
      //   for (int i = 0; i < data.env.params().num_species; i++) {
      //     m_env.get_sub_guard_cells(data.Rho[i]);
      //   }
      // }
      auto &mesh = grid.mesh();
      dim3 blockSize(32, 16);
      dim3 gridSize(mesh.reduced_dim(0) / 32, mesh.reduced_dim(1) / 16);

      Kernels::filter_current_cart<<<gridSize, blockSize>>>(
          get_pitchptr(data.J.data(0)), get_pitchptr(m_tmp_j1),
          m_env.is_boundary(0), m_env.is_boundary(1),
          m_env.is_boundary(2), m_env.is_boundary(3));
      data.J.data(0).copy_from(m_tmp_j1);
      CudaCheckError();

      Kernels::filter_current_cart<<<gridSize, blockSize>>>(
          get_pitchptr(data.J.data(1)), get_pitchptr(m_tmp_j2),
          m_env.is_boundary(0), m_env.is_boundary(1),
          m_env.is_boundary(2), m_env.is_boundary(3));
      data.J.data(1).copy_from(m_tmp_j2);
      CudaCheckError();

      Kernels::filter_current_cart<<<gridSize, blockSize>>>(
          get_pitchptr(data.J.data(2)), get_pitchptr(m_tmp_j2),
          m_env.is_boundary(0), m_env.is_boundary(1),
          m_env.is_boundary(2), m_env.is_boundary(3));
      data.J.data(2).copy_from(m_tmp_j2);
      CudaCheckError();

      if ((step + 1) % data.env.params().data_interval == 0) {
        for (int i = 0; i < data.env.params().num_species; i++) {
          Kernels::filter_current_cart<<<gridSize, blockSize>>>(
              get_pitchptr(data.Rho[i].data()), get_pitchptr(m_tmp_j1),
              m_env.is_boundary(0), m_env.is_boundary(1),
              m_env.is_boundary(2), m_env.is_boundary(3));
          data.Rho[i].data().copy_from(m_tmp_j1);
          CudaCheckError();
        }
      }
      CudaSafeCall(cudaDeviceSynchronize());
    }
    // timer::stamp("ph_update");
    // Skip empty particle array
    if (data.photons.number() > 0) {
      Logger::print_info(
          "Updating {} photons in log spherical coordinates",
          data.photons.number());
      Kernels::move_photons_cart<<<256, 512>>>(
          data.photons.data(), data.photons.number(), dt,
          m_env.is_boundary(2), m_env.is_boundary(3));
      CudaCheckError();
    }
    CudaSafeCall(cudaDeviceSynchronize());
    // timer::show_duration_since_stamp("Updating photons", "us",
    //                                  "ph_update");
  }
  // timer::show_duration_since_stamp("Sending guard cells", "us",
  // "comm");
  // data.send_particles();
  apply_boundary(data, dt, step);
  timer::show_duration_since_stamp("Ptc update", "us", "ptc_update");
}

void
ptc_updater_cart::apply_boundary(sim_data &data, double dt,
                                   uint32_t step) {
  auto data_p = get_data_ptrs(data);
  if (data.env.is_boundary((int)BoundaryPos::lower0)) {
  }
  data.particles.clear_guard_cells(m_env.local_grid());
  data.photons.clear_guard_cells(m_env.local_grid());
  CudaSafeCall(cudaDeviceSynchronize());

  if (data.env.is_boundary((int)BoundaryPos::upper0)) {
    Kernels::ptc_outflow<<<256, 512>>>(data.particles.data(),
                                       data.particles.number());
    CudaCheckError();
  }
  CudaSafeCall(cudaDeviceSynchronize());
}

// void
// ptc_updater_logsph::inject_ptc(sim_data &data, int inj_per_cell,
//                                Scalar p1, Scalar p2, Scalar p3,
//                                Scalar w, Scalar omega) {
//   if (data.env.is_boundary((int)BoundaryPos::lower0)) {
//     m_surface_e.assign_dev(0.0);
//     m_surface_p.assign_dev(0.0);
//     m_surface_tmp.assign_dev(0.0);
//     Kernels::measure_surface_density<<<256, 512>>>(
//         data.particles.data(), data.particles.number(),
//         m_surface_e.dev_ptr(), m_surface_p.dev_ptr());
//     CudaCheckError();
//     Kernels::inject_ptc<<<128, 256>>>(
//         data.particles.data(), data.particles.number(), inj_per_cell,
//         p1, p2, p3, w, m_surface_e.dev_ptr(), m_surface_p.dev_ptr(),
//         (curandState *)data.d_rand_states, omega);
//     CudaCheckError();

//     data.particles.set_num(data.particles.number() +
//                            2 * inj_per_cell *
//                                data.E.grid().mesh().reduced_dim(1));
//   }
// }

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
