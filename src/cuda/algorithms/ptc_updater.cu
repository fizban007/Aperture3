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
// #include "cuda/algorithms/user_push_2d_cart.cuh"

namespace Aperture {

namespace Kernels {

__global__ void
ptc_push_cart(data_ptrs data, size_t num, Scalar dt) {
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
    auto old_x1 = ptc.x1[idx], old_x2 = ptc.x2[idx];
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
          (interp(data.E1, old_x1, old_x2, c1, c2, Stagger(0b110))) *
          q_over_m;
      Scalar E2 =
          (interp(data.E2, old_x1, old_x2, c1, c2, Stagger(0b101))) *
          q_over_m;
      Scalar E3 =
          (interp(data.E3, old_x1, old_x2, c1, c2, Stagger(0b011))) *
          q_over_m;
      Scalar B1 =
          (interp(data.B1, old_x1, old_x2, c1, c2, Stagger(0b001)) +
           interp(data.Bbg1, old_x1, old_x2, c1, c2, Stagger(0b001))) *
          q_over_m;
      Scalar B2 =
          (interp(data.B2, old_x1, old_x2, c1, c2, Stagger(0b010)) +
           interp(data.Bbg2, old_x1, old_x2, c1, c2, Stagger(0b010))) *
          q_over_m;
      Scalar B3 =
          (interp(data.B3, old_x1, old_x2, c1, c2, Stagger(0b100))) *
          q_over_m;

      // printf("B1 = %f, B2 = %f, B3 = %f\n", B1, B2, B3);
      // printf("E1 = %f, E2 = %f, E3 = %f\n", E1, E2, E3);
      // printf("B cell is %f\n", *ptrAddr(fields.B1, c1*sizeof(Scalar)
      // + c2*fields.B1.pitch)); printf("q over m is %f\n", q_over_m);
      // printf("gamma before is %f\n", gamma);
      // printf("p is (%f, %f, %f), gamma is %f\n", p1, p2, p3, gamma);
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
deposit_current_cart_2d(data_ptrs data, size_t num, Scalar dt,
                        uint32_t step) {
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

    Scalar x = dev_mesh.pos(0, c1, old_x1);
    Scalar y = dev_mesh.pos(1, c2, old_x2);
    Scalar z = old_x3;

    v1 = v1 / gamma;
    v2 = v2 / gamma;
    v3 = v3 / gamma;
    // printf("cart position is (%f, %f, %f)\n", x, y, z);
    // printf("new cart position is (%f, %f, %f)\n", x, y, z);

    Pos_t new_x1 = old_x1 + (v1 * dt) / dev_mesh.delta[0];
    Pos_t new_x2 = old_x2 + (v2 * dt) / dev_mesh.delta[1];
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

    ptc.cell[idx] = dev_mesh.get_idx(c1 + dc1, c2 + dc2);
    // printf("new_x1 is %f, new_x2 is %f, dc2 = %d\n", new_x1, new_x2,
    // dc2);
    ptc.x1[idx] = new_x1;
    ptc.x2[idx] = new_x2;
    ptc.x3[idx] = z + v3 * dt;

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
filter_current_cart_2d(pitchptr<Scalar> j, pitchptr<Scalar> j_tmp,
                       bool boundary_lower0, bool boundary_upper0,
                       bool boundary_lower1, bool boundary_upper1) {
  // Load position parameters
  int n1 = dev_mesh.guard[0] + blockIdx.x * blockDim.x + threadIdx.x;
  int n2 = dev_mesh.guard[1] + blockIdx.y * blockDim.y + threadIdx.y;
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
}

__global__ void
filter_current_cart_3d(pitchptr<Scalar> j, pitchptr<Scalar> j_tmp,
                       bool boundary_lower0, bool boundary_upper0,
                       bool boundary_lower1, bool boundary_upper1) {}

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
  Grid &grid = m_env.local_grid();

  timer::stamp("ptc_push");
  // Skip empty particle array
  if (data.particles.number() > 0) {
    Logger::print_info(
        "Updating {} particles in log spherical coordinates",
        data.particles.number());
    if (m_env.grid().dim() == 2) {
      Kernels::ptc_push_cart<<<256, 512>>>(data_p,
                                           data.particles.number(), dt);
    }
    CudaCheckError();
    CudaSafeCall(cudaDeviceSynchronize());

    timer::show_duration_since_stamp("Pushing particles", "us",
                                     "ptc_push");

    timer::stamp("ptc_deposit");

    if (data.particles.number() > 0) {
      if (m_env.grid().dim() == 2) {
        Kernels::deposit_current_cart_2d<<<256, 512>>>(
            data_p, data.particles.number(), dt, step);
      }
      CudaCheckError();
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

      Kernels::filter_current_cart_2d<<<gridSize, blockSize>>>(
          get_pitchptr(data.J, 0), get_pitchptr(m_tmp_j),
          m_env.is_boundary(0), m_env.is_boundary(1),
          m_env.is_boundary(2), m_env.is_boundary(3));
      data.J.data(0).copy_from(m_tmp_j);
      CudaCheckError();

      Kernels::filter_current_cart_2d<<<gridSize, blockSize>>>(
          get_pitchptr(data.J, 1), get_pitchptr(m_tmp_j),
          m_env.is_boundary(0), m_env.is_boundary(1),
          m_env.is_boundary(2), m_env.is_boundary(3));
      data.J.data(1).copy_from(m_tmp_j);
      CudaCheckError();

      Kernels::filter_current_cart_2d<<<gridSize, blockSize>>>(
          get_pitchptr(data.J, 2), get_pitchptr(m_tmp_j),
          m_env.is_boundary(0), m_env.is_boundary(1),
          m_env.is_boundary(2), m_env.is_boundary(3));
      data.J.data(2).copy_from(m_tmp_j);
      CudaCheckError();

      if ((step + 1) % data.env.params().data_interval == 0) {
        for (int i = 0; i < data.env.params().num_species; i++) {
          Kernels::filter_current_cart_2d<<<gridSize, blockSize>>>(
              get_pitchptr(data.Rho[i]), get_pitchptr(m_tmp_j),
              m_env.is_boundary(0), m_env.is_boundary(1),
              m_env.is_boundary(2), m_env.is_boundary(3));
          data.Rho[i].data().copy_from(m_tmp_j);
          CudaCheckError();
        }
      }
      CudaSafeCall(cudaDeviceSynchronize());
    }
  }
  // timer::show_duration_since_stamp("Sending guard cells", "us",
  // "comm");
  // data.send_particles();
  apply_boundary(data, dt, step);
  timer::show_duration_since_stamp("Ptc update", "us", "ptc_update");
}

void
ptc_updater::apply_boundary(sim_data &data, double dt, uint32_t step) {
  data.particles.clear_guard_cells(m_env.local_grid());
  data.photons.clear_guard_cells(m_env.local_grid());
  CudaSafeCall(cudaDeviceSynchronize());

}

}  // namespace Aperture
