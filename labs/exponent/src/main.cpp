#include "cuda/core/sim_environment_dev.h"
#include "cuda/data/array.h"
#include "radiation/spectra.h"
#include "sim.h"
#include "utils/logger.h"
#include "utils/timer.h"
#include "utils/util_functions.h"
#include <random>

using namespace Aperture;

int
main(int argc, char* argv[]) {
  uint32_t step = 0;
  // Construct the simulation environment
  cu_sim_environment env(&argc, &argv);

  exponent_sim sim(env);

  Spectra::black_body ne(1.0e-2);

  sim.init_spectra(ne, 1.0);
  sim.add_new_particles(1, 1.0);
  double dt = env.params().delta_t;
  double Eacc = env.params().constE;

  for (uint32_t step = 0; step < env.params().max_steps; step++) {
    Logger::print_info("==== On timestep {} ====", step);
    sim.push_particles(Eacc, dt);
    sim.produce_pairs();
    sim.produce_photons();

    if (sim.ptc_num > 0.8 * env.params().max_ptc_number ||
        sim.ph_num > 0.8 * env.params().max_photon_number)
      break;
  }

  return 0;
}
