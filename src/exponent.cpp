#include "pic_sim.h"
#include "sim_data.h"
#include "sim_environment.h"
#include "radiation/radiation_transfer.h"
#include "radiation/inverse_compton_power_law.h"
#include "cuda/cudarng.h"
#include "utils/logger.h"
#include "utils/util_functions.h"
#include <iostream>
#include <random>

using namespace Aperture;

int
main(int argc, char *argv[]) {
  Environment env(&argc, &argv);

  // Print the parameters of this run
  Logger::print_info("dt is {}", env.params().delta_t);
  Logger::print_info("E is {}", env.params().constE);
  Logger::print_info("e_min is {}", env.params().e_min);

  // Allocate simulation data
  SimData data(env);

  // Initialize simulator
  PICSim sim(env);
  RadiationTransfer<Particles, Photons,
                    InverseComptonPL1D<Kernels::CudaRng>> rad(env);

  // Initialize seed particle(s)
  int N = 1;
  for (int i = 0; i < N; i++) {
    data.particles.append({env.gen_rand(), 0.0, 0.0}, {0.0, 0.0, 0.0}, 10, ParticleType::electron);
  }
  data.particles.sync_to_device();

  // Main simulation loop
  Scalar dt = env.params().delta_t;
  for (uint32_t step = 0; step < env.params().max_steps; step++) {
    Logger::print_info("At timestep {}", step);
    // if (step % env.params().data_interval == 0) {
      
    // }
    rad.emit_photons(data.photons, data.particles);
    sim.ptc_pusher().push(data, dt);
    rad.produce_pairs(data.particles, data.photons);
  }
  
  return 0;
}
