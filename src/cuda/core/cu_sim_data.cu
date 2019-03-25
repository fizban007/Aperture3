#include "cu_sim_data.h"
#include "cuda/grids/grid_1dgr_dev.h"
#include "cuda/grids/grid_log_sph_dev.h"
#include "cuda/constant_mem.h"
#include "cuda/constant_mem_func.h"
#include "cuda/cudaUtility.h"

namespace Aperture {

namespace Kernels {

__global__ void
fill_particles(particle_data ptc, Scalar weight, int multiplicity) {
  for (int j =
           blockIdx.y * blockDim.y + threadIdx.y + dev_mesh.guard[1];
       j < dev_mesh.dims[1] - dev_mesh.guard[1];
       j += blockDim.y * gridDim.y) {
    for (int i =
             blockIdx.x * blockDim.x + threadIdx.x + dev_mesh.guard[0];
         i < dev_mesh.dims[0] - dev_mesh.guard[0];
         i += blockDim.x * gridDim.x) {
      uint32_t cell = i + j * dev_mesh.dims[0];
      Scalar theta = dev_mesh.pos(1, j, 0.5f);
      // int Np = 3;
      for (int n = 0; n < multiplicity; n++) {
        size_t idx = cell * multiplicity * 2 + n * 2;
        ptc.x1[idx] = ptc.x1[idx + 1] = 0.5f;
        ptc.x2[idx] = ptc.x2[idx + 1] = 0.5f;
        ptc.x3[idx] = ptc.x3[idx + 1] = 0.0f;
        ptc.p1[idx] = ptc.p1[idx + 1] = 0.0f;
        ptc.p2[idx] = ptc.p2[idx + 1] = 0.0f;
        ptc.p3[idx] = ptc.p3[idx + 1] = 0.0f;
        ptc.E[idx] = ptc.E[idx + 1] = 1.0f;
        ptc.cell[idx] = ptc.cell[idx + 1] = cell;
        ptc.weight[idx] = ptc.weight[idx + 1] = weight * sin(theta);
        ptc.flag[idx] = set_ptc_type_flag(0, ParticleType::electron);
        ptc.flag[idx + 1] =
            set_ptc_type_flag(0, ParticleType::positron);
      }
    }
  }
}

}  // namespace Kernels

cu_sim_data::cu_sim_data(const cu_sim_environment& e)
    : env(e), dev_map(e.dev_map()) {
  num_species = env.params().num_species;
  Rho.resize(dev_map.size());
  initialize(e);
}

cu_sim_data::~cu_sim_data() {}

void
cu_sim_data::initialize(const cu_sim_environment& env) {
  init_grid(env);
  for (int n = 0; n < dev_map.size(); n++) {
    // Loop over the devices on the node to initialize each data
    // structure
    int dev_id = dev_map[n];
    CudaSafeCall(cudaSetDevice(dev_id));
    E.emplace_back(*grid[n]);
    E[n].initialize();
    B.emplace_back(*grid[n]);
    B[n].set_field_type(FieldType::B);
    B[n].initialize();
    Ebg.emplace_back(*grid[n]);
    Ebg[n].initialize();
    Bbg.emplace_back(*grid[n]);
    Bbg[n].initialize();
    J.emplace_back(*grid[n]);
    J[n].initialize();
    flux.emplace_back(*grid[n]);
    flux[n].initialize();

    for (int i = 0; i < num_species; i++) {
      Rho[n].emplace_back(*grid[n]);
      Rho[n][i].initialize();
      Rho[n][i].sync_to_host();
    }

    init_dev_bg_fields(Ebg[n], Bbg[n]);
  }

  for (int n = 0; n < dev_map.size(); n++) {
    int dev_id = dev_map[n];
    CudaSafeCall(cudaSetDevice(dev_id));
    cudaDeviceSynchronize();
  }
}

void
cu_sim_data::fill_multiplicity(Scalar weight, int multiplicity) {
  for (int n = 0; n < dev_map.size(); n++) {
    int dev_id = dev_map[n];
    CudaSafeCall(cudaSetDevice(dev_id));
    Kernels::fill_particles<<<dim3(16, 16), dim3(32, 32)>>>(
        particles[n].data(), weight, multiplicity);
    // cudaDeviceSynchronize();
    CudaCheckError();

    auto& mesh = grid[n]->mesh();
    particles[n].set_num(mesh.reduced_dim(0) * mesh.reduced_dim(1) * multiplicity);
  }
}

// void
// cu_sim_data::init_bg_fields() {
//   CudaSafeCall(cudaSetDevice(devId));
//   Ebg = E;
//   Bbg = B;
//   Ebg.sync_to_host();
//   Bbg.sync_to_host();

//   E.assign(0.0);
//   B.assign(0.0);
//   E.sync_to_host();
//   B.sync_to_host();
// }

void
cu_sim_data::init_grid(const cu_sim_environment& env) {
  for (int n = 0; n < dev_map.size(); n++) {
    int dev_id = dev_map[n];
    CudaSafeCall(cudaSetDevice(dev_id));

    // Setup the grids
    if (env.params().coord_system == "Cartesian") {
      grid[n].reset(new Grid());
    } else if (env.params().coord_system == "LogSpherical") {
      grid[n].reset(new Grid_LogSph_dev());
    } else if (env.params().coord_system == "1DGR" && grid[n]->dim() == 1) {
      grid[n].reset(new Grid_1dGR_dev());
    } else {
      grid[n].reset(new Grid());
    }
    grid[n]->init(env.sub_params(n));
    Logger::print_info("Grid dimension for dev {} is {}", dev_id,
                       grid[n]->dim());
    if (grid[n]->mesh().delta[0] < env.params().delta_t) {
      std::cerr
          << "Grid spacing should be larger than delta_t! Aborting!"
          << std::endl;
      abort();
    }
    init_dev_mesh(grid[n]->mesh());
  }
}

}  // namespace Aperture
