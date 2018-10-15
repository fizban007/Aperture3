#include "cuda/cudaUtility.h"
#include "sim_data.h"

namespace Aperture {

// SimData::SimData() :
//     env(Environment::get_instance()) {
//   // const Environment& env = Environment::get_instance();
//   initialize(env);
// }

SimData::SimData(const Environment& e, int deviceId)
    : env(e),
      E(env.local_grid()),
      B(env.local_grid()),
      J(env.local_grid()),
      particles(env.params()),
      photons(env),
      devId(deviceId) {
  // initialize(env);
  num_species = env.params().num_species;
  E.initialize();
  B.initialize();
  J.initialize();

  CudaSafeCall(cudaSetDevice(devId));
  // Initialize the pointer array that contains the device pointers
  // CudaSafeCall(
  //     cudaMallocManaged(&rho_ptrs, num_species * sizeof(Scalar*)));

  for (int i = 0; i < num_species; i++) {
    Rho.emplace_back(env.local_grid());
    Rho[i].sync_to_device();
    // rho_ptrs[i] = Rho[i].ptr();
    Rho_avg.emplace_back(env.local_grid());
    J_s.emplace_back(env.local_grid());
    J_avg.emplace_back(env.local_grid());
    // particles.emplace_back(env.params(),
    // static_cast<ParticleType>(i));
  }

  // particles.sync_to_device(0);
  // photons.sync_to_device(0);
  E.sync_to_device();
  B.sync_to_device();
  J.sync_to_device();
  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
}

SimData::~SimData() {
  // cudaFree(rho_ptrs);
}

void
SimData::initialize(const Environment& env) {}

}  // namespace Aperture
