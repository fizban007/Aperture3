#include "additional_diagnostics.h"
#include "sim_data.h"
#include "sim_environment.h"

namespace Aperture {

namespace Kernels {

__global__ void
collect_diagnostics(particle_data ptc, size_t ptc_num, photon_data photons, size_t ph_num,
                    cudaPitchedPtr ph_per_cell) {}

}

AdditionalDiagnostics::AdditionalDiagnostics(const Environment& env)
    : m_env(env), m_ph_num(env.local_grid()) {
  for (int i = 0; i < m_env.params().num_species; i++) {
    m_gamma.emplace_back(m_env.local_grid());
  }
}

AdditionalDiagnostics::~AdditionalDiagnostics() {}

void
AdditionalDiagnostics::collect_diagnostics(const SimData& data) {
  
}

}  // namespace Aperture