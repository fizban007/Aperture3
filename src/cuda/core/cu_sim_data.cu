#include "cu_sim_data.h"
#include "cuda/constant_mem.h"
#include "cuda/constant_mem_func.h"
#include "cuda/cudaUtility.h"
#include "cuda/grids/grid_1dgr_dev.h"
#include "cuda/grids/grid_log_sph_dev.h"
#include "cuda/utils/iterate_devices.h"

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

__global__ void
send_particles(particle_data ptc_src, size_t num_src,
               particle_data ptc_dst, size_t num_dst, int* ptc_sent,
               int dim, int dir) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < num_src;
       i += blockDim.x * gridDim.x) {
    uint32_t cell = ptc_src.cell[i];
    int c, delta_cell;
    if (dim == 0) {
      c = dev_mesh.get_c1(cell);
      delta_cell = dev_mesh.reduced_dim(0);
    } else if (dim == 1) {
      c = dev_mesh.get_c2(cell);
      delta_cell = dev_mesh.reduced_dim(1) * dev_mesh.dims[0];
    } else if (dim == 2) {
      c = dev_mesh.get_c3(cell);
      delta_cell = dev_mesh.reduced_dim(2) * dev_mesh.dims[0] * dev_mesh.dims[1];
    }
    if ((dir == 0 && c < dev_mesh.guard[dim]) ||
        (dir == 1 && c >= dev_mesh.dims[dim] - dev_mesh.guard[dim])) {
      size_t pos = atomicAdd(ptc_sent, 1) + num_dst;
      ptc_src.cell[i] = MAX_CELL;
      ptc_dst.cell[pos] = cell + (dir == 0 ? 1 : -1) * delta_cell;
      ptc_dst.x1[pos] = ptc_src.x1[i];
      ptc_dst.x2[pos] = ptc_src.x2[i];
      ptc_dst.x3[pos] = ptc_src.x3[i];
      ptc_dst.p1[pos] = ptc_src.p1[i];
      ptc_dst.p2[pos] = ptc_src.p2[i];
      ptc_dst.p3[pos] = ptc_src.p3[i];
      ptc_dst.E[pos] = ptc_src.E[i];
      ptc_dst.weight[pos] = ptc_src.weight[i];
      ptc_dst.flag[pos] = ptc_src.flag[i];
    }
  }
}

__global__ void
send_photons(photon_data ptc_src, size_t num_src, photon_data ptc_dst,
             size_t num_dst, int* ptc_sent, int dim, int dir) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < num_src;
       i += blockDim.x * gridDim.x) {
    uint32_t cell = ptc_src.cell[i];
    int c;
    if (dim == 0) {
      c = dev_mesh.get_c1(cell);
    } else if (dim == 1) {
      c = dev_mesh.get_c2(cell);
    } else if (dim == 2) {
      c = dev_mesh.get_c3(cell);
    }
    if ((dir == 0 && c < dev_mesh.guard[dim]) ||
        (dir == 1 && c >= dev_mesh.dims[dim] - dev_mesh.guard[dim])) {
      size_t pos = atomicAdd(ptc_sent, 1) + num_dst;
      ptc_src.cell[i] = MAX_CELL;
      ptc_dst.cell[pos] = cell;
      ptc_dst.x1[pos] = ptc_src.x1[i];
      ptc_dst.x2[pos] = ptc_src.x2[i];
      ptc_dst.x3[pos] = ptc_src.x3[i];
      ptc_dst.p1[pos] = ptc_src.p1[i];
      ptc_dst.p2[pos] = ptc_src.p2[i];
      ptc_dst.p3[pos] = ptc_src.p3[i];
      ptc_dst.E[pos] = ptc_src.E[i];
      ptc_dst.weight[pos] = ptc_src.weight[i];
      ptc_dst.path_left[pos] = ptc_src.path_left[i];
      ptc_dst.flag[pos] = ptc_src.flag[i];
    }
  }
}

}  // namespace Kernels

cu_sim_data::cu_sim_data(const cu_sim_environment& e)
    : env(e), dev_map(e.dev_map()) {
  num_species = env.params().num_species;
  Rho.resize(num_species);
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
    Logger::print_debug("using device {}", dev_id);
    auto& mesh = grid[n]->mesh();
    Logger::print_debug("Mesh dims are {}x{}x{}", mesh.dims[0],
                        mesh.dims[1], mesh.dims[2]);
    E.emplace_back(*grid[n]);
    // Logger::print_debug("initialize E");
    E[n].initialize();
    B.emplace_back(*grid[n]);
    B[n].set_field_type(FieldType::B);
    // Logger::print_debug("initialize B");
    B[n].initialize();
    Ebg.emplace_back(*grid[n]);
    // Logger::print_debug("initialize Ebg");
    Ebg[n].initialize();
    Bbg.emplace_back(*grid[n]);
    Bbg[n].set_field_type(FieldType::B);
    // Logger::print_debug("initialize Bbg");
    Bbg[n].initialize();
    J.emplace_back(*grid[n]);
    // Logger::print_debug("initialize J");
    J[n].initialize();
    flux.emplace_back(*grid[n]);
    // Logger::print_debug("initialize flux");
    flux[n].initialize();

    for (int i = 0; i < num_species; i++) {
      Rho[i].emplace_back(*grid[n]);
      // Logger::print_debug("initialize Rho[{}]", i);
      Rho[i][n].initialize();
      Rho[i][n].sync_to_host();
    }

    init_dev_bg_fields(Ebg[n], Bbg[n]);

    particles.emplace_back(env.sub_params(n).max_ptc_number);
    photons.emplace_back(env.sub_params(n).max_photon_number);
    // Logger::print_debug("synchronizing the device");
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
    particles[n].set_num(mesh.reduced_dim(0) * mesh.reduced_dim(1) *
                         multiplicity);
  }
}

void
cu_sim_data::init_grid(const cu_sim_environment& env) {
  grid.resize(dev_map.size());
  for (int n = 0; n < dev_map.size(); n++) {
    int dev_id = dev_map[n];
    CudaSafeCall(cudaSetDevice(dev_id));

    // Setup the grids
    if (env.params().coord_system == "Cartesian") {
      grid[n].reset(new Grid());
    } else if (env.params().coord_system == "LogSpherical") {
      grid[n].reset(new Grid_LogSph_dev());
    } else if (env.params().coord_system == "1DGR" &&
               grid[n]->dim() == 1) {
      grid[n].reset(new Grid_1dGR_dev());
    } else {
      grid[n].reset(new Grid());
    }
    grid[n]->init(env.sub_params(n));
    auto& mesh = grid[n]->mesh();
    Logger::print_debug("Grid dimension for dev {} is {}x{}x{}", dev_id,
                        mesh.dims[0], mesh.dims[1], mesh.dims[2]);
    Logger::print_debug("Grid lower are {}, {}, {}", mesh.lower[0],
                        mesh.lower[1], mesh.lower[2]);
    if (grid[n]->mesh().delta[0] < env.params().delta_t) {
      std::cerr
          << "Grid spacing should be larger than delta_t! Aborting!"
          << std::endl;
      abort();
    }
    init_dev_mesh(grid[n]->mesh());
  }
}

void
cu_sim_data::send_particles() {
  std::vector<int*> ptc_sent_left(dev_map.size());
  std::vector<int*> ptc_sent_right(dev_map.size());
  std::vector<int*> ph_sent_left(dev_map.size());
  std::vector<int*> ph_sent_right(dev_map.size());
  for (int i = 0; i < dev_map.size(); i++) {
    CudaSafeCall(cudaSetDevice(dev_map[i]));
    CudaSafeCall(cudaMallocManaged(&(ptc_sent_left[i]), sizeof(int)));
    CudaSafeCall(cudaMallocManaged(&(ptc_sent_right[i]), sizeof(int)));
    CudaSafeCall(cudaMallocManaged(&(ph_sent_left[i]), sizeof(int)));
    CudaSafeCall(cudaMallocManaged(&(ph_sent_right[i]), sizeof(int)));
  }
  for (int i = 0; i < dev_map.size(); i++) {
    if (particles[i].number() == 0) continue;
    CudaSafeCall(cudaSetDevice(dev_map[i]));
    if (i > 0) {
      // Send particles left
      Kernels::send_particles<<<256, 512>>>(
          particles[i].data(), particles[i].number(),
          particles[i - 1].data(), particles[i - 1].number(),
          ptc_sent_left[i], grid[0]->dim() - 1, 0);
      CudaCheckError();
    }
    if (i < dev_map.size() - 1) {
      // Send particles right
      Kernels::send_particles<<<256, 512>>>(
          particles[i].data(), particles[i].number(),
          particles[i + 1].data(), particles[i + 1].number(),
          ptc_sent_right[i], grid[0]->dim() - 1, 1);
      CudaCheckError();
    }
  }
  for (int i = 0; i < dev_map.size(); i++) {
    if (photons[i].number() == 0) continue;
    CudaSafeCall(cudaSetDevice(dev_map[i]));
    if (i > 0) {
      // Send particles left
      Kernels::send_photons<<<256, 512>>>(
          photons[i].data(), photons[i].number(), photons[i - 1].data(),
          photons[i - 1].number(), ph_sent_left[i], grid[0]->dim() - 1,
          0);
      CudaCheckError();
    }
    if (i < dev_map.size() - 1) {
      // Send particles right
      Kernels::send_photons<<<256, 512>>>(
          photons[i].data(), photons[i].number(), photons[i + 1].data(),
          photons[i + 1].number(), ph_sent_right[i], grid[0]->dim() - 1,
          1);
      CudaCheckError();
    }
  }
  for (int i = 0; i < dev_map.size(); i++) {
    CudaSafeCall(cudaSetDevice(dev_map[i]));
    CudaSafeCall(cudaDeviceSynchronize());
    if (i > 0) {
      particles[i - 1].set_num(particles[i - 1].number() +
                               *ptc_sent_left[i]);
      photons[i - 1].set_num(photons[i - 1].number() +
                             *ph_sent_left[i]);
    }
    if (i < dev_map.size() - 1) {
      particles[i + 1].set_num(particles[i + 1].number() +
                               *ptc_sent_right[i]);
      photons[i + 1].set_num(photons[i + 1].number() +
                             *ph_sent_right[i]);
    }
  }
  for (int i = 0; i < dev_map.size(); i++) {
    CudaSafeCall(cudaSetDevice(dev_map[i]));
    CudaSafeCall(cudaFree(ptc_sent_left[i]));
    CudaSafeCall(cudaFree(ptc_sent_right[i]));
    CudaSafeCall(cudaFree(ph_sent_left[i]));
    CudaSafeCall(cudaFree(ph_sent_right[i]));
  }
}

}  // namespace Aperture
