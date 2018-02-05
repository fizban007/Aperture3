#include "sim_data.h"

namespace Aperture {

// SimData::SimData() :
//     env(Environment::get_instance()) {
//   // const Environment& env = Environment::get_instance();
//   initialize(env);
// }

SimData::SimData(const Environment& env) :
    env(env), E(env.local_grid()),
    B(env.local_grid()),
    Bflux(env.local_grid()),
    J(env.local_grid()),
    photons(env) {
  // initialize(env);
  num_species = 3;
  for (int i = 0; i < num_species; i++) {
    Rho.emplace_back(env.local_grid());
    particles.emplace_back(env.conf().max_ptc_number);

    double q = env.conf().q_e;
    if (static_cast<ParticleType>(i) == ParticleType::electron) {
      particles[i].set_charge(-q);
      particles[i].set_mass(1.0);
    } else if (static_cast<ParticleType>(i) == ParticleType::positron) {
      particles[i].set_charge(q);
      particles[i].set_mass(1.0);
    } else if (static_cast<ParticleType>(i) == ParticleType::ion) {
      particles[i].set_charge(q);
      particles[i].set_mass(env.conf().ion_mass);
    }
  }
  for (int i = 0; i < num_species; i++) {
    particles[i].initialize();
  }
}

SimData::~SimData() {}

void
SimData::initialize(const Environment& env) {}

}
