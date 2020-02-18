#ifndef __SIM_DATA_IMPL_H_
#define __SIM_DATA_IMPL_H_

#include "sim_data.h"
#include "sim_environment.h"
#include "utils/timer.h"
#include "utils/logger.h"
#include "core/stagger.h"

namespace Aperture {

sim_data::sim_data(const sim_environment& e)
    : env(e) {
  // Logger::print_info("Particle array size is {}", particles.size());

  E.resize(env.local_grid());
  E.set_field_type(FieldType::E);
  E.initialize();
  Logger::print_info("Local grid extent is {}x{}x{}", env.local_grid().extent().x,
                     env.local_grid().extent().y, env.local_grid().extent().z);

  B.resize(env.local_grid());
  B.set_field_type(FieldType::B);
  B.initialize();

  Bbg.resize(env.local_grid());
  Bbg.set_field_type(FieldType::B);
  Bbg.initialize();

  // J.resize(env.local_grid());
  // J.set_field_type(FieldType::E);
  // J.initialize();

  // divE.resize(env.local_grid());
  // divB.resize(env.local_grid());

  initialize(env);
}

sim_data::~sim_data() {
  finalize();
}

void
sim_data::copy_to_host() {
  Logger::print_info("Sync E");
  E.copy_to_host();
  Logger::print_info("Sync B");
  B.copy_to_host();
  // Logger::print_info("Sync J");
  // J.copy_to_host();
  // Logger::print_info("Sync divE");
  // divE.copy_to_host();
  // Logger::print_info("Sync divB");
  // divB.copy_to_host();
}

}


#endif // __SIM_DATA_IMPL_H_
