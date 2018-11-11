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
                        "data", 2);
  exporter.WriteGrid();

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
  env.B_bg().sync_to_host();

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

  // Initialize a bunch of particles
  // std::default_random_engine gen;
  // std::uniform_int_distribution<int> dist(100, 258);
  // uint32_t N = 1;
  // for (uint32_t i = 0; i < N; i++) {
  //   data.particles.append({0.5f, 0.f, 0.f}, {0.0f, 5.0f, 0.0f},
  //                         // mesh.get_idx(dist(gen), dist(gen)),
  //                         mesh.get_idx(100, 508),
  //                         ParticleType::electron, 1000.0);
  // }
  Logger::print_info("number of particles is {}",
                     data.particles.number());

  for (uint32_t step = 0; step < env.params().max_steps; step++) {
    double dt = env.params().delta_t;
    // double dt = 0.0;
    double time = step * dt;
    Logger::print_info("At timestep {}, time = {}", step, time);

    // Apply boundary conditions
    Scalar omega = 0.0;
    if (time <= 5.0) {
      omega = env.params().omega * (time / 5.0);
    } else {
      omega = env.params().omega;
    }
    field_solver.boundary_conditions(data, omega);

    // Output data
    if ((step % env.params().data_interval) == 0) {
      dynamic_cast<const Grid_LogSph*>(&env.local_grid())
          ->compute_flux(flux, data.B, env.B_bg());
      // Logger::print_info("Finished computing flux");

      // Logger::print_info("Rho 512: {}, 513: {}, 514: {}",
      //                    data.Rho[0](100, 512), data.Rho[0](100,
      //                    513), data.Rho[0](100, 514));
      // Logger::print_info("J2 512: {}, 513: {}, 514: {}",
      //                    data.J(1, 100, 512), data.J(1, 100, 513),
      //                    data.J(1, 100, 514));
      // Logger::print_info("E2 512: {}, 513: {}, 514: {}",
      //                    data.E(1, 100, 512), data.E(1, 100, 513),
      //                    data.E(1, 100, 514));
      // Logger::print_info("divE 512: {}, 513: {}, 514: {}",
      //                    field_solver.get_divE()(100, 512),
      //                    field_solver.get_divE()(100, 513),
      //                    field_solver.get_divE()(100, 514));

      exporter.WriteOutput(step, time);
      exporter.writeXMF(step, time);
      rad.get_ph_events().initialize();
      rad.get_pair_events().initialize();

      Logger::print_info("Finished output!");
    }

    timer::stamp();
    ptc_updater.update_particles(data, dt);
    ptc_updater.handle_boundary(data);
    rad.emit_photons(data.photons, data.particles);
    rad.produce_pairs(data.particles, data.photons);
    if (step % 1 == 0)
      ptc_updater.inject_ptc(data, 1, 0.0, 0.0, 0.0, 100.0, omega);
    auto t_ptc = timer::get_duration_since_stamp("us");
    Logger::print_info("Ptc Update took {}us", t_ptc);

    if (step % 20 == 0) {
      timer::stamp();
      data.particles.sort_by_cell();
      data.photons.sort_by_cell();
      auto t_sort = timer::get_duration_since_stamp("us");
      Logger::print_info("Ptc sort took {}us", t_sort);
    }

    timer::stamp();
    field_solver.update_fields(data.E, data.B, data.J, dt, time);
    auto t_field = timer::get_duration_since_stamp("us");
    Logger::print_info("Field Update took {}us", t_field);
  }
  return 0;
}
