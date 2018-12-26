#include "additional_diagnostics.h"
#include "algorithms/field_solver_log_sph.h"
#include "cuda/constant_mem_func.h"
#include "cuda/cudarng.h"
#include "ptc_updater_logsph.h"
#include "radiation/curvature_instant.h"
// #include "radiation/radiation_transfer.h"
#include "radiation/rt_pulsar.h"
#include "sim_data_dev.h"
#include "sim_environment.h"
#include "utils/logger.h"
#include "utils/timer.h"
#include "utils/util_functions.h"
#include <random>

using namespace Aperture;

int
main(int argc, char* argv[]) {
  uint32_t step = 0;

  // Construct the simulation environment
  Environment env(&argc, &argv);

  // Allocate simulation data
  SimData data(env);

  // Initialize data exporter
  DataExporter exporter(env, data, step);
  
  if (env.params().is_restart) {
    Logger::print_info("This is a restart");
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

  }
  // Initialize the field solver
  FieldSolver_LogSph field_solver(
      *dynamic_cast<const Grid_LogSph*>(&env.grid()));

  // Initialize particle updater
  PtcUpdaterLogSph ptc_updater(env);

  // Initialize radiation module
  RadiationTransferPulsar rad(env);

  // Setup data export and diagnostics
  AdditionalDiagnostics diag(env);
  exporter.AddField("E", data.E);
  exporter.AddField("B", data.B);
  exporter.AddField("B_bg", env.B_bg());
  exporter.AddField("J", data.J);
  exporter.AddField("Rho_e", data.Rho[0]);
  exporter.AddField("Rho_p", data.Rho[1]);
  exporter.AddField("Rho_i", data.Rho[2]);
  exporter.AddField("flux", data.flux, false);
  exporter.AddField("divE", field_solver.get_divE());
  exporter.AddField("divB", field_solver.get_divB());
  exporter.AddField("photon_produced", rad.get_ph_events());
  exporter.AddField("pair_produced", rad.get_pair_events());
  exporter.AddField("photon_num", diag.get_ph_num());
  exporter.AddField("gamma_e", diag.get_gamma(0));
  exporter.AddField("gamma_p", diag.get_gamma(1));
  exporter.AddField("gamma_i", diag.get_gamma(2));
  exporter.AddField("num_e", diag.get_ptc_num(0));
  exporter.AddField("num_p", diag.get_ptc_num(1));
  exporter.AddField("num_i", diag.get_ptc_num(2));

  // Main simulation loop
  for (; step <= env.params().max_steps; step++) {
    double dt = env.params().delta_t;
    double time = step * dt;

    Logger::print_info("=== At timestep {}, time = {} ===", step, time);

    Scalar omega = 0.0;
    if (time <= 10.0) {
      omega = env.params().omega *
              square(std::sin(CONST_PI * 0.5 * (time / 10.0)));
    } else {
      omega = env.params().omega;
    }

    // Output data
    if ((step % env.params().data_interval) == 0) {
      diag.collect_diagnostics(data);
      dynamic_cast<const Grid_LogSph*>(&env.local_grid())
          ->compute_flux(data.flux, data.B, env.B_bg());
      // Logger::print_info("Finished computing flux");

      exporter.WriteOutput(step, time);
      exporter.writeXMF(step, time);
      rad.get_ph_events().initialize();
      rad.get_pair_events().initialize();

      Logger::print_info("Finished output!");
    }

    // Inject particles
    timer::stamp();
    if (step % 1 == 0)
      ptc_updater.inject_ptc(data, 4, 0.0, 0.0, 0.0, 2000.0, omega);

    // Update particles (push and deposit)
    ptc_updater.update_particles(data, dt);
    ptc_updater.handle_boundary(data);
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

    if (step % env.params().sort_interval == 0 && step != 0) {
      timer::stamp();
      data.particles.sort_by_cell();
      data.photons.sort_by_cell();
      auto t_sort = timer::get_duration_since_stamp("us");
      Logger::print_info("Ptc sort took {}us", t_sort);
    }

    if (step % env.params().snapshot_interval == 0 && step > 0) {
      timer::stamp();
      exporter.writeSnapshot(env, data, step);
      auto t_snapshot = timer::get_duration_since_stamp("ms");
      Logger::print_info("Snapshot took {}ms", t_snapshot);
    }
  }
  return 0;
}