#include "cuda/constant_mem.h"
#include "cuda/constant_mem_func.h"
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
    if (cell < dev_mesh.guard[0] ||
        cell > dev_mesh.dims[0] - dev_mesh.guard[0])
      continue;
    // if (cell != 350) continue;
    for (int n = 0; n < multiplicity; n++) {
      size_t idx = cell * multiplicity * 2 + n * 2;
      // ptc.x1[idx] = ptc.x1[idx + 1] = curand_uniform(&local_state);
      ptc.x1[idx] = ptc.x1[idx + 1] = 1.0f;
      ptc.p1[idx] = ptc.p1[idx + 1] = 0.0f;
      Scalar D1 = mesh_ptrs.D1[cell];
      Scalar D2 = mesh_ptrs.D2[cell];
      Scalar D3 = mesh_ptrs.D3[cell];
      Scalar alpha = mesh_ptrs.alpha[cell];
      Scalar K1 = mesh_ptrs.K1[cell];
      Scalar rho0 = mesh_ptrs.rho0[cell];
      Scalar u0 = sqrt(D2 / (D2 * (alpha * alpha - D3) + D1 * D1));

      float u = curand_uniform(&local_state);
      ptc.E[idx] = ptc.E[idx + 1] = u0;
      ptc.cell[idx] = ptc.cell[idx + 1] = cell;
      // ptc.cell[idx] = MAX_CELL;
      ptc.weight[idx] =
          K1 + std::abs(min(rho0 * K1, 0.0f)) / multiplicity;
      ptc.weight[idx + 1] = K1 + max(rho0 * K1, 0.0f) / multiplicity;
      // printf("p1 %f, x1 %f, u0 %f, w %f\n", ptc.p1[idx], ptc.x1[idx],
      // u0, ptc.weight[idx]);
      ptc.flag[idx] = set_ptc_type_flag(
          (u < 0.1 ? bit_or(ParticleFlag::tracked) : 0),
          ParticleType::electron);
      ptc.flag[idx + 1] = set_ptc_type_flag(
          (u < 0.1 ? bit_or(ParticleFlag::tracked) : 0),
          ParticleType::positron);
    }
  }
}

__global__ void
prepare_initial_photons(photon1d_data photons,
                        Grid_1dGR_dev::mesh_ptrs mesh_ptrs,
                        int multiplicity) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  curandState local_state;
  curand_init(1234, id, 0, &local_state);
  for (int cell =
           blockIdx.x * blockDim.x + threadIdx.x + dev_mesh.guard[0];
       cell < dev_mesh.dims[0] - dev_mesh.guard[0];
       cell += blockDim.x * gridDim.x) {
    if (cell < dev_mesh.guard[0] ||
        cell > dev_mesh.dims[0] - dev_mesh.guard[0])
      continue;
    for (int n = 0; n < multiplicity; n++) {
      size_t idx = cell * multiplicity * 2 + n * 2;
      // photons.x1[idx] = photons.x1[idx + 1] =
      // curand_uniform(&local_state);
      photons.x1[idx] = photons.x1[idx + 1] = 1.0f;
      photons.pf[idx] = photons.pf[idx + 1] = 0.0f;
      // TODO: Compute photon E
      Scalar g11 = mesh_ptrs.gamma_rr[cell];
      Scalar g33 = mesh_ptrs.gamma_ff[cell];
      Scalar alpha = mesh_ptrs.alpha[cell];
      constexpr Scalar a = 0.99;
      Scalar xi = dev_mesh.pos(0, cell, 1.0f);
      const Scalar rp = 1.0f + std::sqrt(1.0f - a * a);
      const Scalar rm = 1.0f - std::sqrt(1.0f - a * a);
      Scalar exp_xi = std::exp(xi * (rp - rm));
      Scalar r = (rp - rm * exp_xi) / (1.0 - exp_xi);
      Scalar Delta = r * r - 2.0f * r + a * a;

      photons.p1[idx] = 10.0f * curand_uniform(&local_state) * Delta;
      photons.p1[idx + 1] = -10.0f * curand_uniform(&local_state) * Delta;
      photons.E[idx] = sgn(photons.p1[idx]) *
                       std::sqrt(g11 * square(photons.p1[idx] / Delta) +
                                 g33 * square(photons.pf[idx])) /
                       alpha;
      photons.E[idx + 1] =
          sgn(photons.p1[idx + 1]) *
          std::sqrt(g11 * square(photons.p1[idx + 1] / Delta) +
                    g33 * square(photons.pf[idx + 1])) /
          alpha;

      photons.weight[idx] = photons.weight[idx + 1] = 1.0f;
      photons.cell[idx] = photons.cell[idx + 1] = cell;
      photons.flag[idx] = bit_or(PhotonFlag::tracked);
      photons.flag[idx + 1] = bit_or(PhotonFlag::tracked);
    }
  }
}

}  // namespace Kernels

cu_sim_data1d::cu_sim_data1d(const cu_sim_environment& e)
    : env(e),
      dev_id(e.dev_id()),
      particles(e.params().max_ptc_number),
      photons(e.params().max_photon_number) {
  initialize(e);
}

cu_sim_data1d::~cu_sim_data1d() {}

void
cu_sim_data1d::initialize(const cu_sim_environment& env) {
  // CudaSafeCall(cudaSetDevice(dev_id));
  init_grid(env);
  E = cu_vector_field<Scalar>(*grid);
  E.initialize();
  J = cu_vector_field<Scalar>(*grid);
  J.initialize();

  for (int i = 0; i < env.params().num_species; i++) {
    Rho.emplace_back(*grid);
    Rho[i].initialize();
    // Rho[n][i].sync_to_host();
  }

  Logger::print_debug("Finished initializing sim_data1d");
  cudaDeviceSynchronize();
}

void
cu_sim_data1d::prepare_initial_condition(int multiplicity) {
  const Grid_1dGR_dev* g =
      dynamic_cast<const Grid_1dGR_dev*>(grid.get());
  if (g != nullptr) {
    Kernels::prepare_initial_condition<<<128, 256>>>(
        particles.data(), g->get_mesh_ptrs(), multiplicity);
    CudaCheckError();

    particles.set_num(g->mesh().reduced_dim(0) * 2 * multiplicity);
    // particles.append({0.5, 0.0, 4000})
  }
}

void
cu_sim_data1d::prepare_initial_photons(int multiplicity) {
  const Grid_1dGR_dev* g =
      dynamic_cast<const Grid_1dGR_dev*>(grid.get());
  if (g != nullptr) {
    Kernels::prepare_initial_photons<<<128, 256>>>(
        photons.data(), g->get_mesh_ptrs(), multiplicity);
    CudaCheckError();

    photons.set_num(g->mesh().reduced_dim(0) * 2 * multiplicity);
    // particles.append({0.5, 0.0, 4000})
  }
}

void
cu_sim_data1d::init_grid(const cu_sim_environment& env) {
  if (env.params().coord_system == "Cartesian") {
    grid.reset(new Grid());
  } else if (env.params().coord_system == "1DGR") {
    grid.reset(new Grid_1dGR_dev());
  } else {
    grid.reset(new Grid());
  }
  Logger::print_info("Initializing grid");
  grid->init(env.params());
  Logger::print_info("Grid dimension for dev {} is {}", dev_id,
                     grid->dim());
  if (grid->mesh().delta[0] < env.params().delta_t) {
    std::cerr << "Grid spacing should be larger than delta_t! Aborting!"
              << std::endl;
    abort();
  }
  init_dev_mesh(grid->mesh());
}

}  // namespace Aperture
