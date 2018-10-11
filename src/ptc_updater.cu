#include "cuda/constant_mem.h"
#include "cuda/cudaUtility.h"
#include "data/detail/multi_array_utils.hpp"
#include "ptc_updater.h"
#include "sim_data.h"
#include "sim_environment.h"
#include "utils/logger.h"
#include "utils/util_functions.h"

namespace Aperture {

struct fields_data {
  cudaPitchedPtr E1, E2, E3;
  cudaPitchedPtr B1, B2, B3;
  cudaPitchedPtr J1, J2, J3;
  cudaPitchedPtr* Rho;
};

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
update_particles(particle_data ptc, size_t num, fields_data fields) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num;
       i += blockDim.x * gridDim.x) {
    auto c = ptc.cell[i];
    // Skip empty particles
    if (c == MAX_CELL) continue;

    // step 1: Update particle momentum

    // step 2: Compute particle movement and update position

    // step 3: Deposit current
  }
}

__global__ void
update_particles_1d(particle_data ptc, size_t num, const Scalar* E1,
                    Scalar* J1, Scalar* Rho, Scalar dt) {
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
      s1 = interp(new_x1, cell, c);

      if (!check_bit(flag, ParticleFlag::ignore_current)) {
        s0 = interp(old_x1, cell - delta_cell, c);
        j_sum += dev_charges[sp] * w * (s0 - s1) * dev_mesh.delta[0] /
                 dev_params.delta_t;
        atomicAdd(&J1[c], j_sum);
      }
      atomicAdd(&Rho[c], dev_charges[sp] * w * s1);
    }
  }
}

}  // namespace Kernels

PtcUpdater::PtcUpdater(const Environment& env) : m_env(env) {
  m_extent = m_env.grid().extent();
  // FIXME: select the correct device?
  CudaSafeCall(cudaMallocManaged(
      &m_Rho, m_env.params().num_species * sizeof(cudaPitchedPtr)));
}

PtcUpdater::~PtcUpdater() {
  // FIXME: select the correct device
  CudaSafeCall(cudaFree(m_Rho));
}

void
PtcUpdater::update_particles(SimData& data, double dt) {
  if (m_env.grid().dim() == 1) {
    Kernels::update_particles_1d<<<512, 512>>>(
        data.particles.data(), data.particles.number(),
        (const Scalar*)data.E.ptr(0).ptr, (Scalar*)data.J.ptr(0).ptr,
        (Scalar*)data.Rho[0].ptr().ptr, dt);
    CudaCheckError();
  }
}

void
PtcUpdater::handle_boundary(SimData& data) {}

}  // namespace Aperture
