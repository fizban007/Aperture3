#include "cuda/core/additional_diagnostics.h"
// #include "algorithms/field_solver_log_sph.h"
// #include "cuda/constant_mem_func.h"
// #include "cuda/cudarng.h"
// #include "ptc_updater_logsph.h"
// #include "radiation/curvature_instant.h"
// #include "radiation/radiation_transfer.h"
#include "cuda/core/cu_sim_data.h"
#include "cuda/core/field_solver_log_sph.h"
#include "cuda/core/ptc_updater_logsph.h"
#include "cuda/core/sim_environment_dev.h"
#include "cuda/radiation/rt_magnetar.h"
#include "cuda/utils/cu_data_exporter.h"
#include "utils/logger.h"
#include "utils/timer.h"
#include "utils/util_functions.h"
// #include "core/field_solver_default.h"
#include <random>

using namespace Aperture;

int
main(int argc, char* argv[]) {
  uint32_t step = 0;
  // Construct the simulation environment
  cu_sim_environment env(&argc, &argv);

  // Allocate simulation data
  cu_sim_data data(env);

  // Initialize the field solver
  FieldSolver_LogSph field_solver(
      *dynamic_cast<const Grid_LogSph_dev*>(&env.grid()));
  // Initialize particle updater
  PtcUpdaterLogSph ptc_updater(env);

  // Initialize radiation module
  RadiationTransferMagnetar rad(env);

  // Setup data export and diagnostics
  AdditionalDiagnostics diag(env);

  // Initialize data exporter
  cu_data_exporter exporter(env.params(), step);
  exporter.add_field("E", data.E);
  exporter.add_field("B", data.B);
  exporter.add_field("B_bg", data.Bbg);
  exporter.add_field("J", data.J);
  exporter.add_field("Rho_e", data.Rho[0]);
  exporter.add_field("Rho_p", data.Rho[1]);
  exporter.add_field("Rho_i", data.Rho[2]);
  exporter.add_field("flux", data.flux, false);
  exporter.add_field("divE", field_solver.get_divE());
  exporter.add_field("divB", field_solver.get_divB());
  exporter.add_field("photon_produced", rad.get_ph_events());
  exporter.add_field("pair_produced", rad.get_pair_events());
  exporter.add_field("photon_num", diag.get_ph_num());
  exporter.add_field("gamma_e", diag.get_gamma(0));
  exporter.add_field("gamma_p", diag.get_gamma(1));
  exporter.add_field("gamma_i", diag.get_gamma(2));
  exporter.add_field("num_e", diag.get_ptc_num(0));
  exporter.add_field("num_p", diag.get_ptc_num(1));
  exporter.add_field("num_i", diag.get_ptc_num(2));
  exporter.add_array_output("ph_flux", data.photon_flux);
  exporter.add_ptc_output_2d("particle", "ptc", &data.particles);

  if (env.params().is_restart) {
    Logger::print_info("======= THIS IS A RESTART =======");
    exporter.load_from_snapshot(env, data, step);
    exporter.prepareXMFrestart(step, env.params().data_interval);
    step += 1;
  } else {
    exporter.copyConfigFile();
    exporter.copySrc();
    exporter.WriteGrid();

    // Setup initial conditions
    Scalar B0 = env.params().B0;
    auto& mesh = env.grid().mesh();
    data.E.initialize();
    data.B.initialize();
    data.B.initialize(0, [B0, mesh](Scalar x1, Scalar x2, Scalar x3) {
      Scalar r = exp(x1);
      // return 2.0 * B0 * cos(x2) / (r * r * r);
      return B0 * cos(x2) *
             (1.0 / square(exp(x1 - 0.5 * mesh.delta[0])) -
              1.0 / square(exp(x1 + 0.5 * mesh.delta[0]))) /
             (r * mesh.delta[0]);
    });
    data.B.initialize(1, [B0, mesh](Scalar x1, Scalar x2, Scalar x3) {
      Scalar r = exp(x1);
      // return B0 * sin(x2) / (r * r * r);
      return B0 *
             (cos(x2 - 0.5 * mesh.delta[1]) -
              cos(x2 + 0.5 * mesh.delta[1])) /
             (r * r * r * mesh.delta[1]);
    });
    data.B.sync_to_device();
    // Put the initial condition to the background
    env.init_bg_fields(data);

    // for (int j = mesh.guard[1]; j < mesh.dims[1] - mesh.guard[1]; j++) {
    //   uint32_t cell = 5 + j * mesh.dims[0];
    //   data.particles.append({0.5, 0.5, 0.0}, {1000.0, 0.0, 0.0}, cell,
    //                         ParticleType::electron, 1000.0);
    // }
    data.set_initial_condition(100.0, 2);
  }

  for (; step <= env.params().max_steps; step++) {
    double dt = env.params().delta_t;
    double time = step * dt;

    Logger::print_info("=== At timestep {}, time = {} ===", step, time);

    Scalar omega = 0.0;
    if (time <= 30.0) {
      omega = env.params().omega * (time / 30.0);
    } else if (time <= 60.0) {
      omega = env.params().omega * (1.0 - (time - 30.0) / 30.0);
    } else {
      omega = 0.0;
    }

    // Output data
    if ((step % env.params().data_interval) == 0) {
      diag.collect_diagnostics(data);
      cudaDeviceSynchronize();
      dynamic_cast<const Grid_LogSph_dev*>(&env.local_grid())
          ->compute_flux(data.flux, data.B, data.Bbg);
      // Logger::print_info("Finished computing flux");

      exporter.WriteOutput(step, time);
      exporter.write_particles(step, time);
      exporter.writeXMF(step, time);
      rad.get_ph_events().initialize();
      rad.get_pair_events().initialize();

      Logger::print_info("Finished output!");
    }

    timer::stamp();
    // Inject particles
    if (step % 4 == 0)
      ptc_updater.inject_ptc(data, 1, 0.0, 0.0, 0.0, 100.0, omega);
    // Update particles (push and deposit)
    ptc_updater.update_particles(data, dt, step);
    ptc_updater.handle_boundary(data);
    if (step % 20 == 0) {
      ptc_updater.annihilate_extra_pairs(data, dt);
    }
    auto t_ptc = timer::get_duration_since_stamp("us");
    Logger::print_info("Ptc Update took {}us", t_ptc);

    // Update field values and handle field boundary conditions
    timer::stamp();
    field_solver.update_fields(data.E, data.B, data.J, dt, time);
    field_solver.boundary_conditions(data, omega);

    auto t_field = timer::get_duration_since_stamp("us");
    Logger::print_info("Field Update took {}us", t_field);

    // Create photons and pairs
    rad.emit_photons(data);
    rad.produce_pairs(data);

    // Sort the particles every once in a while
    if (step % env.params().sort_interval == 0 && step != 0) {
      timer::stamp();
      data.particles.sort_by_cell();
      data.photons.sort_by_cell();
      auto t_sort = timer::get_duration_since_stamp("us");
      Logger::print_info("Ptc sort took {}us", t_sort);
    }

    // Write a snapshot at given interval
    if (step % env.params().snapshot_interval == 0 && step > 0) {
      timer::stamp();
      exporter.write_snapshot(env, data, step);
      auto t_snapshot = timer::get_duration_since_stamp("ms");
      Logger::print_info("Snapshot took {}ms", t_snapshot);
    }
  }

  return 0;
}
