#include "cuda/constant_mem.h"
#include "cuda/core/cu_sim_data1d.h"
#include "cuda/cudaUtility.h"
#include "cuda/grids/grid_1dgr_dev.h"
#include "data/particle_data.h"
#include <curand_kernel.h>

namespace Aperture {

namespace Kernels {

__global__ void
prepare_initial_condition(particle1d_data ptc,
                          Grid_1dGR_dev::mesh_ptrs mesh_ptrs,
                          int multiplicity) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  curandState local_state;
  curand_init(1234, id, 0, &local_state);
  for (int cell =
           blockIdx.x * blockDim.x + threadIdx.x + dev_mesh.guard[0];
       cell < dev_mesh.dims[0] - dev_mesh.guard[0];
       cell += blockDim.x * gridDim.x) {
    if (cell < dev_mesh.guard[0] || cell > dev_mesh.dims[0] - dev_mesh.guard[0])
      continue;
    // if (cell != 350) continue;
    for (int n = 0; n < multiplicity; n++) {
      size_t idx = cell * multiplicity * 2 + n * 2;
      ptc.x1[idx] = ptc.x1[idx + 1] = curand_uniform(&local_state);
      // ptc.x1[idx] = ptc.x1[idx + 1] = 0.5f;
      ptc.p1[idx] = ptc.p1[idx + 1] = 0.0f;
      Scalar D1 = mesh_ptrs.D1[cell];
      Scalar D2 = mesh_ptrs.D2[cell];
      Scalar D3 = mesh_ptrs.D3[cell];
      Scalar alpha = mesh_ptrs.alpha[cell];
      Scalar dpsidth = mesh_ptrs.dpsidth[cell];
      Scalar rho0 = mesh_ptrs.rho0[cell];
      Scalar u0 = sqrt(D2 / (D2 * (alpha * alpha - D3) + D1 * D1));

      ptc.E[idx] = ptc.E[idx + 1] = u0;
      ptc.cell[idx] = ptc.cell[idx + 1] = cell;
      // ptc.cell[idx] = MAX_CELL;
      ptc.weight[idx] = 1.0f / dpsidth + std::abs(min(rho0 / dpsidth, 0.0f)) / multiplicity;
      ptc.weight[idx + 1] = 1.0f / dpsidth + max(rho0 / dpsidth, 0.0f) / multiplicity;
      ptc.flag[idx] = set_ptc_type_flag(bit_or(ParticleFlag::tracked), ParticleType::electron);
      ptc.flag[idx + 1] = set_ptc_type_flag(bit_or(ParticleFlag::tracked), ParticleType::positron);
    }
  }
}

}  // namespace Kernels

cu_sim_data1d::cu_sim_data1d(const cu_sim_environment& e, int deviceId)
    : env(e),
      E(env.local_grid()),
      // B(env.local_grid()),
      J(env.local_grid()),
      particles(env.params().max_ptc_number),
      photons(env.params().max_photon_number),
      devId(deviceId) {
  E.initialize();
  // B.initialize();
  J.initialize();

  for (int i = 0; i < env.params().num_species; i++) {
    Rho.emplace_back(env.local_grid());
    Rho[i].initialize();
    Rho[i].sync_to_host();
  }

  E.sync_to_host();
  // B.sync_to_host();
  J.sync_to_host();

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  Logger::print_info("Each particle is worth {} bytes",
                     particle1d_data::size);
  Logger::print_info("Each photon is worth {} bytes",
                     photon1d_data::size);
}

cu_sim_data1d::~cu_sim_data1d() {}

void
cu_sim_data1d::initialize(const cu_sim_environment& env) {}

void
cu_sim_data1d::prepare_initial_condition(int multiplicity) {
  const Grid_1dGR_dev* g =
      dynamic_cast<const Grid_1dGR_dev*>(&env.local_grid());
  if (g != nullptr) {
    Kernels::prepare_initial_condition<<<128, 256>>>(
        particles.data(), g->get_mesh_ptrs(), multiplicity);
    CudaCheckError();

    particles.set_num(g->mesh().reduced_dim(0) * 2 * multiplicity);
    // particles.append({0.5, 0.0, 4000})
  }
}

}  // namespace Aperture