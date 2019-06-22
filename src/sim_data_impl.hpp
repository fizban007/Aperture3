#ifndef __SIM_DATA_IMPL_H_
#define __SIM_DATA_IMPL_H_

#include "sim_data.h"
#include "sim_environment.h"
#include "utils/timer.h"
#include "core/stagger.h"

namespace Aperture {

sim_data::sim_data(const sim_environment& e)
    : env(e),
      particles(env.params().max_ptc_number),
      photons(env.params().max_photon_number) {
  num_species = env.params().num_species;

  E.resize(env.local_grid());
  E.set_field_type(FieldType::E);
  Ebg.resize(env.local_grid());
  E.set_field_type(FieldType::E);

  B.resize(env.local_grid());
  B.set_field_type(FieldType::B);
  Bbg.resize(env.local_grid());
  Bbg.set_field_type(FieldType::B);

  J.resize(env.local_grid());
  J.set_field_type(FieldType::E);
  for (int i = 0; i < num_species; i++) {
    Rho.emplace_back(env.local_grid());
    Rho[i].set_stagger(0, Stagger(0b111));
    gamma.emplace_back(env.local_grid());
    gamma[i].set_stagger(0, Stagger(0b111));
    ptc_num.emplace_back(env.local_grid());
    ptc_num[i].set_stagger(0, Stagger(0b111));
  }

  flux.resize(env.local_grid());
  divE.resize(env.local_grid());
  divB.resize(env.local_grid());
  EdotB.resize(env.local_grid());
  photon_produced.resize(env.local_grid());
  pair_produced.resize(env.local_grid());
  photon_num.resize(env.local_grid());

  initialize(env);
}

sim_data::~sim_data() {
  finalize();
}

void
sim_data::sort_particles() {
  timer::stamp("ptc_sort");
  particles.sort_by_cell(env.local_grid());
  photons.sort_by_cell(env.local_grid());
  timer::show_duration_since_stamp("Sorting particles", "us",
                                   "ptc_sort");
}


}


#endif // __SIM_DATA_IMPL_H_
