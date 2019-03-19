#include "sim_data.h"
#include "sim_environment.h"
#include "grids/grid_log_sph.h"

namespace Aperture {

sim_data::sim_data(const sim_environment& e)
    : env(e),
      particles(env.params().max_ptc_number),
      photons(env.params().max_photon_number) {
  num_species = env.params().num_species;
  initialize(env);
}

sim_data::~sim_data() {}

void
sim_data::initialize(const sim_environment& env) {
  init_grid(env);
  E.resize(*grid);
  B.resize(*grid);
  J.resize(*grid);
  flux.resize(*grid);
  for (int i = 0; i < num_species; i++) {
    Rho.emplace_back(*grid);
  }
}

void
sim_data::init_grid(const sim_environment& env) {
  // Setup the grid
  if (env.params().coord_system == "Cartesian") {
    grid.reset(new Grid());
  } else if (env.params().coord_system == "LogSpherical") {
    grid.reset(new Grid_LogSph());
  } else {
    grid.reset(new Grid());
  }
  grid->init(env.params());
}

}  // namespace Aperture
