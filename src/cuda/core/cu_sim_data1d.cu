#include "cuda/core/cu_sim_data1d.h"
#include "cuda/cudaUtility.h"

namespace Aperture {

cu_sim_data1d::cu_sim_data1d(const cu_sim_environment& e, int deviceId)
    : env(e),
      E(env.local_grid()),
      B(env.local_grid()),
      J(env.local_grid()),
    particles(env.params().max_ptc_number),
    devId(deviceId) {
  E.initialize();
  B.initialize();
  J.initialize();

  for (int i = 0; i < env.params().num_species; i++) {
    Rho.emplace_back(env.local_grid());
    Rho[i].initialize();
    Rho[i].sync_to_host();
  }

  E.sync_to_host();
  B.sync_to_host();
  J.sync_to_host();
  
  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
}

cu_sim_data1d::~cu_sim_data1d() {}

void
cu_sim_data1d::initialize(const cu_sim_environment& env) {}

}