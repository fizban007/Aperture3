// #include "cuda/core/cu_sim_data1d.h"
#include "sim_data.h"
#include "sim_environment.h"
#include "algorithms/field_solver_1dgr.h"
#include "algorithms/ptc_updater_1dgr.h"
#include "radiation/radiative_transfer.h"
// #include "cuda/radiation/rt_1dgr.h"
#include "utils/data_exporter.h"
#include "utils/logger.h"
#include "utils/timer.h"
#include "utils/util_functions.h"
#include <random>

using namespace Aperture;

int
main(int argc, char* argv[]) {
  // We declare the current timestep here because we might need to
  // restart from a snapshot
  uint32_t step = 0;

  // Construct the simulation environment
  sim_environment env(&argc, &argv);

  // Allocate simulation data
  sim_data data(env);

  // Initialize the field solver
  field_solver_1dgr solver(env);
  Logger::print_debug("Finished initializing field solver");

  // Initialize particle updater
  ptc_updater_1dgr pusher(env);
  Logger::print_debug("Finished initializing ptc updater");

  // Initialize radiative transfer module
  // RadiationTransfer1DGR rad(env);
  radiative_transfer rad(env);
  Logger::print_debug("Finished initializing radiation module");

  // Initialize particle distribution in the beginning
  pusher.prepare_initial_condition(data, 20);
  // data.prepare_initial_photons(1);
  Logger::print_debug("Finished initializing initial condition");

  // Initialize data exporter
  // cu_data_exporter exporter(env.params(), step);
  data_exporter exporter(env, step);

  // exporter.add_field("E", data.E);
  // exporter.add_field("J", data.J);
  // exporter.add_field("Rho_e", data.Rho[0]);
  // exporter.add_field("Rho_p", data.Rho[1]);
  // exporter.add_ptc_output_1d("particles", "ptc", &data.particles);
  // exporter.add_ptc_output_1d("photons", "photon", &data.photons);

  // exporter.copyConfigFile();
  exporter.write_grid();

  // Main simulation loop
  for (; step <= env.params().max_steps; step++) {
    double dt = env.params().delta_t;
    double time = step * dt;
    Logger::print_info("=== At timestep {}, time = {} ===", step, time);

    // Output data
    if (step % env.params().data_interval == 0) {
      Logger::print_info("Writing output");
      exporter.write_output(data, step, time);
      // exporter.write_particles(data, step, time);
      // exporter.writeXMF(step, time);
    }

    timer::stamp();
    pusher.update_particles(data, dt);
    data.particles.clear_guard_cells(env.grid());
    data.photons.clear_guard_cells(env.grid());
    auto t_ptc = timer::get_duration_since_stamp("us");
    Logger::print_info("Updating {} particles took {}us",
                       data.particles.number(), t_ptc);

    solver.update_fields(data, dt);

    rad.emit_photons(data);
    rad.produce_pairs(data);

    // Sort the particles every once in a while
    if (step % env.params().sort_interval == 0 && step != 0) {
      timer::stamp();
      data.particles.sort_by_cell(env.grid());
      data.photons.sort_by_cell(env.grid());
      auto t_sort = timer::get_duration_since_stamp("us");
      Logger::print_info("Ptc sort took {}us", t_sort);
    }
  }

  return 0;
}
