#include "sim_data.h"
#include "sim_environment.h"

namespace Aperture {

sim_data::sim_data(const sim_environment& e)
    : env(e),
      E(env.local_grid()),
      B(env.local_grid()),
      J(env.local_grid()),
      flux(env.local_grid()),
      particles(env.params().max_ptc_number),
      photons(env.params().max_photon_number) {
  B.set_field_type(FieldType::B);

  num_species = env.params().num_species;
  for (int i = 0; i < num_species; i++) {
    Rho.emplace_back(env.local_grid());
  }
}

sim_data::~sim_data() {}

void
sim_data::initialize(const sim_environment& env) {}

}  // namespace Aperture
