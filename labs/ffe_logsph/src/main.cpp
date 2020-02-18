#include "cuda/constant_mem_func.h"
#include "algorithms/ffe_solver_logsph.h"
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

  // Allocate simulation data
  sim_data data(env);

  // Initialize data exporter
  data_exporter exporter(env, start_step);

  exporter.copy_config_file();
  exporter.write_grid();

  // Setup initial conditions
  Scalar B0 = env.params().B0;
  Logger::print_debug("B0 in main is {}", B0);
  Logger::print_debug("max tracked is {}", MAX_TRACKED);

  // auto& mesh = env.grid().mesh();
  data.init_bg_B_field(0, [B0](Scalar x1, Scalar x2, Scalar x3) {
    Scalar r = exp(x1);
    return star_field_b1(r, x2, x3);
    // return B0 / (r * r);
  });
  data.init_bg_B_field(1, [B0](Scalar x1, Scalar x2, Scalar x3) {
    Scalar r = exp(x1);
    return star_field_b2(r, x2, x3);
    // return 0.0;
  });
  data.init_bg_B_field(
      2, [B0](Scalar x1, Scalar x2, Scalar x3) { return 0.0; });

  // Initialize the field solver
  ffe_solver_logsph field_solver(env);
  
  // Main simulation loop
  for (; step <= env.params().max_steps; step++) {
    double dt = env.params().delta_t;
    time = start_time + (step - start_step) * dt;

    Scalar omega = env.params().omega;
    // Scalar atm_time = 0.0;
    // Scalar sp_time = 0.2;
    // if (time <= atm_time) {
    //   omega = 0.0;
    // } else if (time <= atm_time + sp_time) {
    //   // omega = env.params().omega *
    //   //         square(std::sin(CONST_PI * 0.5 * (time / 10.0)));
    //   omega = env.params().omega * ((time - atm_time) / sp_time);
    // }
    // else {
    //   omega = env.params().omega;
    // }
    Logger::print_info("=== At timestep {}, time = {}, omega = {} ===",
                       step, time, omega);

    // Output data
    if ((step % env.params().data_interval) == 0) {
      exporter.write_output(data, step, time);
    }

    field_solver.update_fields(data, dt, omega, time);
    // field_solver.apply_boundary(data, time, omega);
  }
  return 0;
}
