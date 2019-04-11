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

  // Spectra::black_body ne(2.0e-5);
  // sim.init_spectra(ne, 1.334e28 / 5.86e-4);
  Spectra::black_body ne(2.0e-5);
  sim.init_spectra(ne, 1.02e31 / 5.86e-4);

  sim.add_new_particles(1, 1.0);
  double dt = env.params().delta_t;
  double Eacc = env.params().constE;
  Logger::print_debug("dt is {}, Eacc is {}", dt, Eacc);

  for (uint32_t step = 0; step < env.params().max_steps; step++) {
    Logger::print_info("==== On timestep {}, pushing {} particles, {} photons ====", step, sim.ptc_num, sim.ph_num);
    // sim.ptc_E.sync_to_host();
    // Logger::print_info("0th particle has energy {}", sim.ptc_E[0]);
    sim.push_particles(Eacc, dt);
    sim.produce_pairs(dt);
    sim.produce_photons(dt);
    sim.compute_spectrum();

    // if (step % env.params().sort_interval == 0 && step != 0) {
    //   sim.sort_photons();
    // }

    if (sim.ptc_num > 0.8 * env.params().max_ptc_number ||
        sim.ph_num > 0.8 * env.params().max_photon_number)
      break;
  }

  return 0;
}
