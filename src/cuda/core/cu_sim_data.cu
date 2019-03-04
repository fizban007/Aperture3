#include "cu_sim_data.h"
#include "cuda/constant_mem.h"
#include "cuda/constant_mem_func.h"
#include "cuda/cudaUtility.h"

namespace Aperture {

namespace Kernels {

__global__ void
fill_particles(particle_data ptc, Scalar weight) {
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
      int Np = 3;
      for (int n = 0; n < Np; n++) {
        size_t idx = cell * Np * 2 + n * 2;
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
        ptc.flag[idx + 1] = set_ptc_type_flag(0, ParticleType::positron);
      }
    }
  }
}

}  // namespace Kernels

// cu_sim_data::cu_sim_data() :
//     env(cu_sim_environment::get_instance()) {
//   // const cu_sim_environment& env =
//   cu_sim_environment::get_instance(); initialize(env);
// }

cu_sim_data::cu_sim_data(const cu_sim_environment& e, int deviceId)
    : env(e),
      E(env.local_grid()),
      B(env.local_grid()),
      J(env.local_grid()),
      Bbg(env.local_grid()),
      Ebg(env.local_grid()),
      flux(env.local_grid()),
      particles(env.params()),
      photons(env.params()),
      devId(deviceId) {
  // initialize(env);
  CudaSafeCall(cudaSetDevice(devId));

  num_species = env.params().num_species;
  B.set_field_type(FieldType::B);
  E.initialize();
  B.initialize();
  Ebg.initialize();
  Bbg.initialize();
  J.initialize();
  flux.initialize();

  for (int i = 0; i < num_species; i++) {
    Rho.emplace_back(env.local_grid());
    Rho[i].initialize();
    Rho[i].sync_to_host();
    // Rho_avg.emplace_back(env.local_grid());
    // J_s.emplace_back(env.local_grid());
    // J_avg.emplace_back(env.local_grid());
    // particles.emplace_back(env.params(),
    // static_cast<ParticleType>(i));
  }

  E.sync_to_host();
  B.sync_to_host();
  Ebg.sync_to_host();
  Bbg.sync_to_host();
  J.sync_to_host();
  flux.sync_to_host();
  init_dev_bg_fields(Ebg, Bbg);
  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
}

cu_sim_data::~cu_sim_data() {}

void
cu_sim_data::initialize(const cu_sim_environment& env) {}

void
cu_sim_data::set_initial_condition(Scalar weight) {
  CudaSafeCall(cudaSetDevice(devId));
  Kernels::fill_particles<<<dim3(16, 16), dim3(32, 32)>>>(particles.data(), weight);
  cudaDeviceSynchronize();
  CudaCheckError();

  auto& mesh = E.grid().mesh();
  particles.set_num(mesh.reduced_dim(0) * mesh.reduced_dim(1) * 6);
}

void
cu_sim_data::init_bg_fields() {
  CudaSafeCall(cudaSetDevice(devId));
  Ebg = E;
  Bbg = B;
  Ebg.sync_to_host();
  Bbg.sync_to_host();

  E.assign(0.0);
  B.assign(0.0);
  E.sync_to_host();
  B.sync_to_host();
}

}  // namespace Aperture
