#include "cuda_runtime.h"
#include "cuda/constant_mem_func.h"
#include "cuda/core/additional_diagnostics.h"
#include "cuda/core/cu_sim_data.h"
#include "cuda/core/cu_sim_environment.h"
#include "cuda/core/field_solver_log_sph.h"
#include "cuda/core/ptc_updater_logsph.h"
#include "cuda/cudarng.h"
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
  float time = 0.0;

  // Construct the simulation environment
  cu_sim_environment env(&argc, &argv);

  // Allocate simulation data
  cu_sim_data data(env);

  // Initialize data exporter
  cu_data_exporter exporter(env.params(), step);

  if (env.params().is_restart) {
    Logger::print_info("This is a restart");
    exporter.load_from_snapshot(env, data, step, time);
    exporter.prepareXMFrestart(step, env.params().data_interval, time);
    step += 1;
    time += env.params().delta_t;
  } else {
    exporter.copyConfigFile();
    exporter.copySrc();
    exporter.WriteGrid();

    // Setup initial conditions
    Scalar B0 = env.params().B0;
    Logger::print_debug("B0 in main is {}", B0);
    // auto& mesh = env.grid().mesh();
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

    // data.particles[0].append({0.5, 0.5, 0.0}, {0.0, 30.0, 0.0}, 50 +
    // 260*129,
    //                          ParticleType::electron, 100.0);
    // data.fill_multiplicity(1.0, 2);
  }
  // Initialize the field solver
  FieldSolver_LogSph field_solver;

  // Initialize particle updater
  PtcUpdaterLogSph ptc_updater(env);

  // Initialize radiation module
  RadiationTransferPulsar rad(env);

  // Setup data export and diagnostics
  AdditionalDiagnostics diag(env);
  exporter.prepare_output(data);

  // Main simulation loop
  for (; step <= env.params().max_steps; step++) {
    double dt = env.params().delta_t;

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
        data.EdotB[n].initialize();
      });

      Logger::print_info("Finished output!");
    }

    // Inject particles
    if (step % 1 == 0)
      ptc_updater.inject_ptc(data, 1, 0.0, 0.0, 0.0, 1.0, omega);

    // Update particles (push and deposit, and handle boundary)
    ptc_updater.update_particles(data, dt, step);

    // Update field values and handle field boundary conditions
    field_solver.update_fields(data, dt, time);
    field_solver.boundary_conditions(data, omega);

    // Create photons and pairs
    rad.emit_photons(data);
    rad.produce_pairs(data);

    if (step % env.params().sort_interval == 0 && step != 0) {
      data.sort_particles();
    }

    if (step % env.params().snapshot_interval == 0 && step > 0) {
      timer::stamp();
      exporter.write_snapshot(env, data, step, time);
      auto t_snapshot = timer::get_duration_since_stamp("ms");
      Logger::print_info("Snapshot took {}ms", t_snapshot);
    }

    time += dt;
  }
  return 0;
}
