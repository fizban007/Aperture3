#ifndef __SIM_DATA_IMPL_H_
#define __SIM_DATA_IMPL_H_

#include "sim_data.h"
#include "sim_environment.h"
#include "utils/timer.h"
#include "utils/logger.h"
#include "core/stagger.h"

namespace Aperture {

sim_data::sim_data(const sim_environment& e)
    : env(e),
      particles(env.params().max_ptc_number),
      photons(env.params().max_photon_number) {
  num_species = env.params().num_species;
  // Logger::print_info("Particle array size is {}", particles.size());

  E.resize(env.local_grid());
  E.set_field_type(FieldType::E);
  E.initialize();
  Logger::print_info("Local grid extent is {}x{}x{}", env.local_grid().extent().x,
                     env.local_grid().extent().y, env.local_grid().extent().z);

  Ebg.resize(env.local_grid());
  Ebg.set_field_type(FieldType::E);
  Ebg.initialize();

  B.resize(env.local_grid());
  B.set_field_type(FieldType::B);
  B.initialize();

  Bbg.resize(env.local_grid());
  Bbg.set_field_type(FieldType::B);
  Bbg.initialize();

  J.resize(env.local_grid());
  J.set_field_type(FieldType::E);
  J.initialize();

  ph_flux = multi_array<Scalar>(Extent(200, 256));
  ph_flux.assign_dev(0.0f);
  ph_flux.copy_to_host();

  Rho.resize(num_species);
  gamma.resize(num_species);
  ptc_num.resize(num_species);
  for (int i = 0; i < num_species; i++) {
    Rho[i] = scalar_field<Scalar>(env.local_grid());
    Rho[i].initialize();
    Rho[i].set_stagger(0, Stagger(0b111));
    gamma[i] = scalar_field<Scalar>(env.local_grid());
    gamma[i].initialize();
    gamma[i].set_stagger(0, Stagger(0b111));
    ptc_num[i] = scalar_field<Scalar>(env.local_grid());
    ptc_num[i].initialize();
    ptc_num[i].set_stagger(0, Stagger(0b111));
  }

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

void
sim_data::copy_to_host() {
  Logger::print_info("Sync E");
  E.copy_to_host();
  Logger::print_info("Sync B");
  B.copy_to_host();
  Logger::print_info("Sync J");
  J.copy_to_host();
  Logger::print_info("Sync rho");
  for (int n = 0; n < num_species; n++)
    Rho[n].copy_to_host();
  Logger::print_info("Sync gamma");
  for (int n = 0; n < num_species; n++)
    gamma[n].copy_to_host();
  Logger::print_info("Sync ptc_num");
  for (int n = 0; n < num_species; n++)
    ptc_num[n].copy_to_host();
  Logger::print_info("Sync divE");
  divE.copy_to_host();
  Logger::print_info("Sync divB");
  divB.copy_to_host();
  Logger::print_info("Sync EdotB");
  EdotB.copy_to_host();
  Logger::print_info("Sync photon_produced");
  photon_produced.copy_to_host();
  Logger::print_info("Sync pair_produced");
  pair_produced.copy_to_host();
  Logger::print_info("Sync photon_num");
  photon_num.copy_to_host();
  Logger::print_info("Sync ph_flux");
  ph_flux.copy_to_host();
}

}


#endif // __SIM_DATA_IMPL_H_
