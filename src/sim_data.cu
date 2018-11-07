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
      photons(env.params()),
      devId(deviceId) {
  // initialize(env);
  num_species = env.params().num_species;
  B.set_field_type(FieldType::B);
  E.initialize();
  B.initialize();
  J.initialize();

  CudaSafeCall(cudaSetDevice(devId));

  for (int i = 0; i < num_species; i++) {
    Rho.emplace_back(env.local_grid());
    Rho[i].initialize();
    Rho[i].sync_to_device();
    // Rho_avg.emplace_back(env.local_grid());
    // J_s.emplace_back(env.local_grid());
    // J_avg.emplace_back(env.local_grid());
    // particles.emplace_back(env.params(),
    // static_cast<ParticleType>(i));
  }

  E.sync_to_device();
  B.sync_to_device();
  J.sync_to_device();
  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
}

SimData::~SimData() {
}

void
SimData::initialize(const Environment& env) {}

}  // namespace Aperture
