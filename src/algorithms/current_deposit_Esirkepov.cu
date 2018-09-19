#include "algorithms/current_deposit_Esirkepov.h"
#include "cuda/constant_mem.h"
#include "cuda/cudaUtility.h"
#include "data/detail/multi_array_utils.hpp"
#include "sim_environment.h"
#include "utils/util_functions.h"
#include <iostream>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

namespace Aperture {

namespace Kernels {

HD_INLINE Scalar
cloud_in_cell(Scalar dx) {
  return max(1.0 - std::abs(dx), 0.0);
}

HD_INLINE Scalar
interp(Scalar rel_pos, int ptc_cell, int target_cell) {
  Scalar dx =
      ((Scalar)target_cell + 0.5 - (rel_pos + Scalar(ptc_cell)));
  return cloud_in_cell(dx);
}

__global__ void
compute_delta_rho_1d(Scalar** rho, Scalar** delta_rho,
                     particle_data ptc, Grid::const_mesh_ptrs mp,
                     uint32_t num) {
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num;
       i += blockDim.x * gridDim.x) {
    auto c = ptc.cell[i];
    // Skip empty particles
    if (c == MAX_CELL) continue;
    int c_p = c;
    auto x = ptc.x1[i];
    // auto x_p = x - ptc.dx1[i];
    auto x_p = x;
    auto flag = ptc.flag[i];
    int sp = get_ptc_type(flag);
    auto q = dev_charges[sp];
    auto w = ptc.weight[i];
    // if (i == 0) printf("q is %f, w is %f\n", q, w);

    // c_p and x_p represent previous location of the particle
    c_p += floor(x_p);
    x_p -= (Scalar)c_p - c;

    Scalar s0, s1;
    for (int delta_c = c_p - 2; delta_c <= c_p + 2; delta_c++) {
      // if (i == 0) printf("x is %f, c is %d, c_p is %d, delta_c is %d,
      // s1 is %f\n", x, c, c_p, delta_c, s1);
      s1 = interp(x, c, delta_c);

      if (!check_bit(flag, ParticleFlag::ignore_current)) {
        s0 = interp(x_p, c_p, delta_c);
        atomicAdd(&delta_rho[sp][delta_c],
                  q * w * (s0 - s1) * mp.A[delta_c] *
                      dev_mesh.delta[0] / dev_params.delta_t);
      }
      atomicAdd(&rho[sp][delta_c], q * w * s1 * mp.A[delta_c]);
      // if (i == 0) printf("rho is %f\n", rho[sp][delta_c]);
    }
  }
}

}  // namespace Kernels

CurrentDepositer_Esirkepov::CurrentDepositer_Esirkepov(
    const Environment& env)
    : m_env(env) {}

CurrentDepositer_Esirkepov::~CurrentDepositer_Esirkepov() {}

void
CurrentDepositer_Esirkepov::deposit(SimData& data, double dt) {
  Logger::print_detail("Depositing current");
  auto& part = data.particles;
  auto& grid = data.E.grid();
  data.J.initialize();

  Scalar** rho_ptrs;
  CudaSafeCall(
      cudaMallocManaged(&rho_ptrs, data.num_species * sizeof(Scalar*)));

  for (Index_t i = 0; i < data.num_species; i++) {
    data.Rho[i].initialize();
    rho_ptrs[i] = data.Rho[i].data().data();
    // data.J_s[i].initialize();
    // data.V[i].initialize();
    // compute_delta_rho(data.J_s[i], data.Rho[i], part[i], dt);
    // normalize_density(data.Rho[i], data.Rho[i]);
  }
  // Scalar** rho_ptrs = data.rho_ptrs;
  Scalar** j_ptrs = data.J.array_ptrs();

  if (m_env.grid().dim() == 1) {
    Kernels::compute_delta_rho_1d<<<512, 512>>>(
        rho_ptrs, j_ptrs, part.data(), grid.get_mesh_ptrs(),
        part.number());
    CudaCheckError();
    cudaDeviceSynchronize();
  }

  // TODO::Handle periodic boundary by copying over the deposited
  // quantities

  for (Index_t i = 0; i < data.num_species; i++) {
    Logger::print_debug("rho at 10 is {}, rhoptr at 10 is {}",
                        data.Rho[i](10), rho_ptrs[i][10]);
  }
  cudaFree(rho_ptrs);

  // communication on the just deposited Rho
  if (m_comm_rho != nullptr) {
    for (Index_t i = 0; i < data.num_species; i++) {
      m_comm_rho(data.Rho[i]);
    }
  }

  // Now we have delta Q in every cell, add them up along all directions
  scan_current(data.J);
  // Call communication on just scanned J
  if (m_comm_J != nullptr) {
    m_comm_J(data.J);
  }

  // TODO::periodic boundary issues
}

void
CurrentDepositer_Esirkepov::compute_delta_rho(
    vfield& J, sfield& Rho, const Particles& particles, double dt) {
  auto& part = particles.data();
  auto& grid = J.grid();
  if (grid.dim() == 1) {
  }
}

void
CurrentDepositer_Esirkepov::compute_delta_rho(
    sfield& J, sfield& Rho, const Particles& particles, double dt) {
  auto& part = particles.data();
  auto& grid = J.grid();
  if (grid.dim() == 1) {
  }
}

void
CurrentDepositer_Esirkepov::scan_current(sfield& J) {
  auto& grid = J.grid();
  if (grid.dim() == 1) {
    // In place scan
    // Logger::print_info("Scanning current");
    auto j_ptr = thrust::device_pointer_cast(J.data().data());
    thrust::inclusive_scan(j_ptr, j_ptr + grid.mesh().dims[0], j_ptr);
    CudaCheckError();
  }
}

void
CurrentDepositer_Esirkepov::scan_current(vfield& J) {
  auto& grid = J.grid();
  if (grid.dim() == 1) {
    // In place scan
    Logger::print_info("Scanning current");
    auto j_ptr = thrust::device_pointer_cast(J.data(0).data());
    thrust::inclusive_scan(j_ptr, j_ptr + grid.mesh().dims[0], j_ptr);
    CudaCheckError();
    Logger::print_debug("last j is {}", J(0, grid.mesh().dims[0] - 1));
  }
}

// FIXME: Boundary conditions!
void
CurrentDepositer_Esirkepov::normalize_current(const vfield& I,
                                              vfield& J) {
  auto& grid = I.grid();
  auto& mesh = I.grid().mesh();
  if (grid.dim() == 1) {
    for (int i = 0; i < mesh.dims[0]; i++) {
      J.data(0)[i] = I.data(0)[i] * mesh.delta[0];
    }
  }
}

void
CurrentDepositer_Esirkepov::normalize_density(const sfield& Q,
                                              sfield& rho, sfield& V) {}

void
CurrentDepositer_Esirkepov::normalize_velocity(const sfield& rho,
                                               sfield& V) {
  auto& grid = rho.grid();
  auto& mesh = grid.mesh();
  if (grid.dim() == 1) {
    for (int i = 0; i < mesh.dims[0]; i++) {
      // J.data(0)[i] = I.data(0)[i] * mesh.delta[0];
      if (std::abs(rho(i)) > 1e-5)
        V(i) /= rho(i);
      else
        V(i) = 0.0;
    }
  }
}

}  // namespace Aperture
