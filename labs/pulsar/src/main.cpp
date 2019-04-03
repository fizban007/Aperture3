#include "cuda/constant_mem_func.h"
#include "cuda/core/additional_diagnostics.h"
#include "cuda/core/field_solver_log_sph.h"
#include "cuda/core/ptc_updater_logsph.h"
#include "cuda/cudarng.h"
// #include "cuda/radiation/curvature_instant.h"
// #include "radiation/radiation_transfer.h"
#include "cuda/core/cu_sim_data.h"
#include "cuda/core/sim_environment_dev.h"
#include "cuda/radiation/rt_pulsar.h"
#include "cuda/utils/cu_data_exporter.h"
#include "cuda/utils/iterate_devices.h"
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

  // Allocate simulation data
  cu_sim_data data(env);

  // Initialize data exporter
  cu_data_exporter exporter(env.params(), step);

  // if (env.params().is_restart) {
  //   Logger::print_info("This is a restart");
  //   exporter.load_from_snapshot(env, data, step);
  //   exporter.prepareXMFrestart(step, env.params().data_interval);
  //   step += 1;
  // } else {
  exporter.copyConfigFile();
  exporter.copySrc();
  exporter.WriteGrid();

  // Setup initial conditions
  Scalar B0 = env.params().B0;
  Logger::print_debug("B0 in main is {}", B0);
  auto& mesh = env.grid().mesh();
  data.init_bg_B_field(0, [B0](Scalar x1, Scalar x2, Scalar x3) {
    Scalar r = exp(x1);
    return 2.0 * B0 * cos(x2) / (r * r * r);
  });
  data.init_bg_B_field(1, [B0](Scalar x1, Scalar x2, Scalar x3) {
    Scalar r = exp(x1);
    return B0 * sin(x2) / (r * r * r);
  });
  data.init_bg_B_field(
      2, [B0](Scalar x1, Scalar x2, Scalar x3) { return 0.0; });
  data.init_bg_fields();

  // for (int j = mesh.guard[1]; j < mesh.dims[1] - mesh.guard[1];
  // j++) {
  //   for (int i = mesh.guard[0]; i < mesh.dims[0] - mesh.guard[0];
  //        i++) {
  //     uint32_t cell = i + j * mesh.dims[0];
  //     for (int n = 0; n < 5; n++) {
  //       data.particles.append({0.5, 0.5, 0.0}, {0.0, 0.0, 0.0},
  //       cell,
  //                             ParticleType::electron, 100.0);
  //       data.particles.append({0.5, 0.5, 0.0}, {0.0, 0.0, 0.0},
  //       cell,
  //                             ParticleType::positron, 100.0);
  //     }
  //   }
  // }
  // data.particles[0].append({0.5, 0.5, 0.0}, {0.0, 30.0, 0.0}, 50 + 260*129,
  //                          ParticleType::electron, 100.0);
  // data.fill_multiplicity(20.0, 2);
  // Initialize the field solver
  FieldSolver_LogSph field_solver;

  // Initialize particle updater
  PtcUpdaterLogSph ptc_updater(env);

  // Initialize radiation module
  RadiationTransferPulsar rad(env);

  // Setup data export and diagnostics
  AdditionalDiagnostics diag(env);
  exporter.prepare_output(data);
  // exporter.add_field("E", data.E);
  // exporter.add_field("B", data.B);
  // exporter.add_field("B_bg", data.Bbg);
  // exporter.add_field("J", data.J);
  // exporter.add_field("Rho_e", data.Rho[0]);
  // exporter.add_field("Rho_p", data.Rho[1]);
  // exporter.add_field("Rho_i", data.Rho[2]);
  // exporter.add_field("flux", data.flux, false);
  // exporter.add_field("divE", field_solver.get_divE());
  // exporter.add_field("divB", field_solver.get_divB());
  // exporter.add_field("photon_produced", rad.get_ph_events());
  // exporter.add_field("pair_produced", rad.get_pair_events());
  // exporter.add_field("photon_num", diag.get_ph_num());
  // exporter.add_field("gamma_e", diag.get_gamma(0));
  // exporter.add_field("gamma_p", diag.get_gamma(1));
  // exporter.add_field("gamma_i", diag.get_gamma(2));
  // exporter.add_field("num_e", diag.get_ptc_num(0));
  // exporter.add_field("num_p", diag.get_ptc_num(1));
  // exporter.add_field("num_i", diag.get_ptc_num(2));

  // Main simulation loop
  for (; step <= env.params().max_steps; step++) {
    double dt = env.params().delta_t;
    double time = step * dt;

    Scalar omega = 0.0;
    if (time <= 10.0) {
      // omega = env.params().omega *
      //         square(std::sin(CONST_PI * 0.5 * (time / 10.0)));
      omega = env.params().omega * (time / 10.0);
    } else {
      omega = env.params().omega;
    }
    Logger::print_info("=== At timestep {}, time = {}, omega = {} ===",
                       step, time, omega);

    // Output data
    if ((step % env.params().data_interval) == 0) {
      diag.collect_diagnostics(data);
      // dynamic_cast<const Grid_LogSph_dev*>(&env.local_grid())
      //     ->compute_flux(data.flux, data.B, data.Bbg);
      // Logger::print_info("Finished computing flux");

      exporter.write_output(data, step, time);
      exporter.writeXMF(step, time);
      for_each_device(env.dev_map(), [&data](int n) {
        data.photon_produced[n].initialize();
        data.pair_produced[n].initialize();
      });

      Logger::print_info("Finished output!");
    }

    // Inject particles
    if (step % 1 == 0)
      ptc_updater.inject_ptc(data, 1, 0.0, 0.0, 0.0, 1.0, omega);

    // Update particles (push and deposit)
    ptc_updater.update_particles(data, dt, step);
    // ptc_updater.handle_boundary(data);

    // Update field values and handle field boundary conditions
    field_solver.update_fields(data, dt, time);
    field_solver.boundary_conditions(data, omega);

    // Create photons and pairs
    rad.emit_photons(data);
    rad.produce_pairs(data);

    if (step % env.params().sort_interval == 0 && step != 0) {
      data.sort_particles();
    }

    // if (step % env.params().snapshot_interval == 0 && step > 0) {
    //   timer::stamp();
    //   exporter.write_snapshot(env, data, step);
    //   auto t_snapshot = timer::get_duration_since_stamp("ms");
    //   Logger::print_info("Snapshot took {}ms", t_snapshot);
    // }
  }
  return 0;
}
