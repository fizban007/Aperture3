#include "sim_data.h"

namespace Aperture {

// SimData::SimData() :
//     env(Environment::get_instance()) {
//   // const Environment& env = Environment::get_instance();
//   initialize(env);
// }

SimData::SimData(const Environment& e) :
    env(e), E(env.local_grid()),
    B(env.local_grid()),
    J(env.local_grid()),
    photons(env) {
  // initialize(env);
  num_species = 3;
  E.initialize();
  B.initialize();
  J.initialize();
  for (int i = 0; i < num_species; i++) {
    Rho.emplace_back(env.local_grid());
    Rho_avg.emplace_back(env.local_grid());
    J_s.emplace_back(env.local_grid());
    J_avg.emplace_back(env.local_grid());
    particles.emplace_back(env.conf().data(), static_cast<ParticleType>(i));
  }
  for (int i = 0; i < num_species; i++) {
    particles[i].initialize();
  }
}

SimData::~SimData() {}

void
SimData::initialize(const Environment& env) {}

}
