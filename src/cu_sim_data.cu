#include "cuda/cudaUtility.h"
#include "cu_sim_data.h"
#include "cuda/constant_mem_func.h"

namespace Aperture {

// cu_sim_data::cu_sim_data() :
//     env(Environment::get_instance()) {
//   // const Environment& env = Environment::get_instance();
//   initialize(env);
// }

cu_sim_data::cu_sim_data(const Environment& e, int deviceId)
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

cu_sim_data::~cu_sim_data() {
}

void
cu_sim_data::initialize(const Environment& env) {}

}  // namespace Aperture
