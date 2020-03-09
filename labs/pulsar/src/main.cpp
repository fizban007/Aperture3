#include "cuda/constant_mem_func.h"
// #include "cuda/core/additional_diagnostics.h"
#include "algorithms/field_solver_logsph.h"
#include "algorithms/ptc_updater_logsph.h"
#include "radiation/radiative_transfer.h"
#include "sim_data.h"
#include "sim_environment.h"
#include "utils/data_exporter.h"
#include "utils/logger.h"
#include "utils/timer.h"
#include "utils/util_functions.h"
#include <random>
#include <cuda_runtime.h>
#include "cuda/cudaUtility.h"

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

  // Allocate simulation data
  sim_data data(env);

  // Initialize data exporter
  data_exporter exporter(env, start_step);

  if (params.is_restart) {
    Logger::print_info("This is a restart");
    exporter.load_snapshot(params.restart_file, data, start_step,
                           start_time);
    step = start_step + 1;
    time = start_time + env.params().delta_t;

    cudaDeviceSynchronize();
    CudaCheckError();
  } else {
    // Setup initial conditions
    Scalar B0 = env.params().B0;
    Logger::print_debug("B0 in main is {}", B0);
    // auto& mesh = env.grid().mesh();
    data.Bbg.initialize(0, [B0](Scalar x1, Scalar x2, Scalar x3) {
      Scalar r = exp(x1);
      return 2.0 * B0 * cos(x2) / (r * r * r);
      // return B0 / (r * r);
    });
    data.Bbg.initialize(1, [B0](Scalar x1, Scalar x2, Scalar x3) {
      Scalar r = exp(x1);
      return B0 * sin(x2) / (r * r * r);
      // return 0.0;
    });
    data.init_bg_fields();

    data.fill_multiplicity(1.0, 5);
  }

  // Initialize the field solver
  field_solver_logsph field_solver(env);

  // Initialize particle updater
  ptc_updater_logsph ptc_updater(env);

  // Initialize radiation module
  radiative_transfer rad(env);

  // Setup data export and diagnostics
  // AdditionalDiagnostics diag(env);
  // exporter.prepare_output(data);

  // Main simulation loop
  for (; step <= env.params().max_steps; step++) {
    double dt = env.params().delta_t;
    time = start_time + (step - start_step) * dt;

    Scalar omega = 0.0;
    Scalar atm_time = 0.0;
    Scalar sp_time = 10.0;
    if (time <= atm_time) {
      omega = 0.0;
    } else if (time <= atm_time + sp_time) {
      // omega = env.params().omega *
      //         square(std::sin(CONST_PI * 0.5 * (time / 10.0)));
      omega = env.params().omega * ((time - atm_time) / sp_time);
    } else {
      omega = env.params().omega;
    }
    Logger::print_info("=== At timestep {}, time = {}, omega = {} ===",
                       step, time, omega);

    // Output data
    if ((step % env.params().data_interval) == 0) {
      // diag.collect_diagnostics(data);
      // dynamic_cast<const Grid_LogSph_dev*>(&env.local_grid())
      //     ->compute_flux(data.flux, data.B, data.Bbg);
      // Logger::print_info("Finished computing flux");

      exporter.write_output(data, step, time);
      // exporter.writeXMF(step, time);
      data.photon_produced.initialize();
      data.pair_produced.initialize();
      data.EdotB.initialize();

      Logger::print_info("Finished output!");
    }

    // Inject particles
    // if (env.params().inject_particles && step % 2 == 0)
    //   ptc_updater.inject_ptc(
    //       data, 4, 0.02, 0.0, 0.0,
    //       // 1.1 * omega / env.params().omega, omega);
    //       1.0, omega);

    // Update particles (push and deposit, and handle boundary)
    ptc_updater.update_particles(data, dt, step);

    // Update field values and handle field boundary conditions
    if (env.params().update_fields) {
      field_solver.update_fields(data, dt, time);
      field_solver.apply_boundary(data, omega);
    }

    // Create photons and pairs
    if (env.params().create_pairs) {
      rad.emit_photons(data);
      rad.produce_pairs(data);
    }

    if (step % env.params().sort_interval == 0 && step != 0) {
      data.sort_particles();
    }

    if (step % env.params().snapshot_interval == 0 && step > 0) {
      timer::stamp();
      exporter.save_snapshot(params.data_dir + "snapshot.h5", data,
                             step, time);
      auto t_snapshot = timer::get_duration_since_stamp("ms");
      Logger::print_info("Snapshot took {}ms", t_snapshot);
    }
  }
  return 0;
}
