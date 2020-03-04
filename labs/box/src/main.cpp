#include "algorithms/field_solver.h"
#include "algorithms/ptc_updater.h"
#include "sim_data.h"
#include "sim_environment.h"
#include "utils/data_exporter.h"
#include "utils/logger.h"
#include "utils/timer.h"
#include "utils/util_functions.h"
#include <random>

using namespace Aperture;

int
main(int argc, char* argv[]) {
  uint32_t start_step = 0;
  uint32_t step = start_step;
  float start_time = 0.0;
  float time = start_time;

  // Construct the simulation environment
  sim_environment env(&argc, &argv);
  auto& params = env.params();
  auto& grid = env.local_grid();

  // Initialize simulation data
  sim_data data(env);

  // Initialize algorithm modules
  field_solver solver(env);
  ptc_updater pusher(env);
  data_exporter exporter(env, step);

  // Setup initial condition
  Scalar kx, ky, ex_norm, ey_norm, exy_norm;
  kx = 1.0 * 2.0 * M_PI / params.size[0];
  ky = 2.0 * 2.0 * M_PI / params.size[1];
  if (ky != 0) {
    ex_norm = 1.0;
    ey_norm = -kx / ky;
  } else {
    ey_norm = 1.0;
    ex_norm = -ky / kx;
  }
  exy_norm = std::sqrt(ex_norm * ex_norm + ey_norm * ey_norm);
  ex_norm = ex_norm / exy_norm;
  ey_norm = ey_norm / exy_norm;

  data.E.initialize(0, [&](Scalar x, Scalar y, Scalar z) {
    // Put your initial condition for Ex here
    return ex_norm * std::sin(kx * x + ky * y);
  });

  data.E.initialize(1, [&](Scalar x, Scalar y, Scalar z) {
    // Put your initial condition for Ey here
    return ey_norm * std::sin(kx * x + ky * y);
  });

  data.B.initialize(2, [&](Scalar x, Scalar y, Scalar z) {
    // Put your initial condition for Bz here
    return std::sin(kx * x + ky * y);
  });

  env.send_guard_cells(data.E);
  env.send_guard_cells(data.B);

  data.fill_multiplicity(1.0f, 10);

  // Main pic loop
  for (; step <= params.max_steps; step++) {
    Scalar dt = params.delta_t;
    time = start_time + (step - start_step) * dt;
    Logger::print_info("=== At timestep {}, time = {} ===",
                       step, time);

    if ((step % params.data_interval) == 0) {
      exporter.write_output(data, step, time);
      data.EdotB.initialize();
    }

    // Update particles (push and deposit, and handle boundary)
    pusher.update_particles(data, dt, step);

    // Update field values and handle field boundary conditions
    if (env.params().update_fields) {
      solver.update_fields(data, dt, step);
    }

    // Sort particles when necessary to help cache locality
    if (step % env.params().sort_interval == 0 && step != 0) {
      data.sort_particles();
    }
  }

  return 0;
}
