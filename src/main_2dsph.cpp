#include "additional_diagnostics.h"
#include "algorithms/field_solver_log_sph.h"
#include "cuda/constant_mem_func.h"
#include "cuda/cudarng.h"
#include "ptc_updater_logsph.h"
#include "radiation/curvature_instant.h"
// #include "radiation/radiation_transfer.h"
#include "radiation/rt_pulsar.h"
#include "sim_data.h"
#include "sim_environment.h"
#include "utils/logger.h"
#include "utils/timer.h"
#include "utils/util_functions.h"
#include <random>

using namespace Aperture;

int
main(int argc, char* argv[]) {
  // Construct the simulation environment
  Environment env(&argc, &argv);

  // Allocate simulation data
  SimData data(env);

  // Initialize the field solver
  FieldSolver_LogSph field_solver(
      *dynamic_cast<const Grid_LogSph*>(&env.grid()));

  // Initialize particle updater
  PtcUpdaterLogSph ptc_updater(env);

  // Initialize radiation module
  RadiationTransferPulsar rad(env);

  // Initialize data exporter
  DataExporter exporter(env.params(),
                        env.params().data_dir + "2d_weak_pulsar",
                        "data", env.params().downsample);
  Logger::print_info("{}", env.params().downsample);
  exporter.WriteGrid();
  AdditionalDiagnostics diag(env);

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

  ScalarField<Scalar> flux(env.grid());
  flux.initialize();
  exporter.AddField("E", data.E);
  exporter.AddField("B", data.B);
  exporter.AddField("B_bg", env.B_bg());
  exporter.AddField("J", data.J);
  exporter.AddField("Rho_e", data.Rho[0]);
  exporter.AddField("Rho_p", data.Rho[1]);
  exporter.AddField("flux", flux, false);
  exporter.AddField("divE", field_solver.get_divE());
  exporter.AddField("divB", field_solver.get_divB());
  exporter.AddField("photon_produced", rad.get_ph_events());
  exporter.AddField("pair_produced", rad.get_pair_events());
  exporter.AddField("photon_num", diag.get_ph_num());
  exporter.AddField("gamma_e", diag.get_gamma(0));
  exporter.AddField("gamma_p", diag.get_gamma(1));
  exporter.AddField("num_e", diag.get_ptc_num(0));
  exporter.AddField("num_p", diag.get_ptc_num(1));

  // Initialize a bunch of particles
  std::default_random_engine gen;
  std::uniform_int_distribution<int> dist(200, 300);
  std::uniform_real_distribution<float> dist_f(0.0, 1.0);
  uint32_t N = 0;
  for (uint32_t i = 0; i < N; i++) {
    data.particles.append({0.5f, 0.5f, 0.f}, {0.0f, 100.0f, 0.0f},
                          // mesh.get_idx(dist(gen), dist(gen)),
                          mesh.get_idx(4, 512), ParticleType::electron,
                          1000.0);
    // }
    // for (uint32_t i = 0; i < N; i++) {
    data.particles.append({0.5f, 0.5f, 0.f}, {0.0f, 0.0f, 0.0f},
                          // mesh.get_idx(dist(gen), dist(gen)),
                          mesh.get_idx(4, 512), ParticleType::positron,
                          1000.0);
  }
  Logger::print_info("number of particles is {}",
                     data.particles.number());

  for (uint32_t step = 0; step < env.params().max_steps; step++) {
    double dt = env.params().delta_t;
    // double dt = 0.0;
    double time = step * dt;
    Logger::print_info("At timestep {}, time = {}", step, time);

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
          ->compute_flux(flux, data.B, env.B_bg());
      // Logger::print_info("Finished computing flux");

      exporter.WriteOutput(step, time);
      exporter.writeXMF(step, time);
      rad.get_ph_events().initialize();
      rad.get_pair_events().initialize();

      Logger::print_info("Finished output!");
    }

    // Inject particles
    timer::stamp();
    if (step % 5 == 0)
      ptc_updater.inject_ptc(data, 1, 0.0, 0.0, 0.0, 400.0, omega);

    ptc_updater.update_particles(data, dt);
    ptc_updater.handle_boundary(data);
    auto t_ptc = timer::get_duration_since_stamp("us");
    Logger::print_info("Ptc Update took {}us", t_ptc);

    timer::stamp();
    field_solver.update_fields(data.E, data.B, data.J, dt, time);

    field_solver.boundary_conditions(data, omega);
    // field_solver.boundary_conditions(data, 0.0);
    auto t_field = timer::get_duration_since_stamp("us");
    Logger::print_info("Field Update took {}us", t_field);

    // Create photons and pairs
    rad.emit_photons(data);
    rad.produce_pairs(data);

    if (step % env.params().sort_frequency == 0 && step != 0) {
      timer::stamp();
      data.particles.sort_by_cell();
      data.photons.sort_by_cell();
      auto t_sort = timer::get_duration_since_stamp("us");
      Logger::print_info("Ptc sort took {}us", t_sort);
    }

    // if (step == 1) {
    //   int c1 = dist(gen);
    //   int c2 = dist(gen);
    //   float x1 = dist_f(gen);
    //   float x2 = dist_f(gen);
    //   for (int n = 0; n < 1; n++) {
    //     data.photons.append({x1, x2, 0.0f}, {1.0f, 1.0f, 0.0f}, -0.01,
    //                         mesh.get_idx(c1, c2), 10000.0);
    //     // data.photons.append({x1, x2, 0.0f}, {-100.0f, -100.0f, 0.0f},
    //     //                     -0.01, mesh.get_idx(c1, c2), 100.0);
    //   }
    // }
    // if (step == 100 || step == 200) {
    //   int c1 = dist(gen);
    //   int c2 = dist(gen);
    //   float x1 = dist_f(gen);
    //   float x2 = dist_f(gen);
    //   for (int n = 0; n < 1000; n++) {
    //     data.particles.append(
    //         {x1, x2, 0.0f}, {100.0f, 100.0f, 0.0f},
    //         mesh.get_idx(c1, c2), ParticleType::electron, 100.0);
    //     data.particles.append(
    //         {x1, x2, 0.0f}, {-100.0f, -100.0f, 0.0f},
    //         mesh.get_idx(c1, c2), ParticleType::positron, 100.0);
    //   }
    // }
  }
  return 0;
}
