#include "algorithms/field_solver_log_sph.h"
#include "ptc_updater_logsph.h"
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

  // Initialize data exporter
  DataExporter exporter(env.params(),
                        "/home/alex/storage/Data/Aperture3/2d_test/",
                        "data", 1);
  exporter.WriteGrid();

  Scalar B0 = 20000.0;
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
  exporter.AddField("flux", flux);
  exporter.AddField("divE", field_solver.get_divE());
  exporter.AddField("divB", field_solver.get_divB());

  // Initialize a bunch of particles
  std::default_random_engine gen;
  std::uniform_int_distribution<int> dist(100, 258);
  uint32_t N = 1;
  for (uint32_t i = 0; i < N; i++) {
    data.particles.append({0.f, 0.f, 0.f}, {0.0f, -5.0f, 0.0f},
                          // mesh.get_idx(dist(gen), dist(gen)),
                          mesh.get_idx(100, 10),
                          ParticleType::electron, 1.0);
  }
  Logger::print_info("number of particles is {}",
                     data.particles.number());
  // data.particles.sync_to_device();

  // exporter.WriteOutput(0, 0.0);
  // exporter.writeXMF(0, 0.0);

  for (uint32_t step = 0; step < env.params().max_steps; step++) {
    double dt = env.params().delta_t;
    // double dt = 0.0;
    double time = step * dt;
    Logger::print_info("At timestep {}, time = {}", step, time);

    // Apply boundary conditions
    if (time <= 5.0) {
      field_solver.boundary_conditions(data, 0.1*(time / 5.0));
    } else {
      field_solver.boundary_conditions(data, 0.1);
    }

    if ((step % env.params().data_interval) == 0) {
      data.E.sync_to_host();
      data.B.sync_to_host();
      data.J.sync_to_host();
      data.Rho[0].sync_to_host();
      data.Rho[1].sync_to_host();
      dynamic_cast<const Grid_LogSph*>(&env.local_grid())
          ->compute_flux(flux, data.B, env.B_bg());
      Logger::print_info("Finished computing flux");
      field_solver.get_divE().sync_to_host();
      field_solver.get_divB().sync_to_host();

      exporter.WriteOutput(step, time);
      exporter.writeXMF(step, time);
    }

    timer::stamp();
    ptc_updater.update_particles(data, dt);
    ptc_updater.handle_boundary(data);
    // if (step == 0)
    ptc_updater.inject_ptc(data, 2, 10.0, 0.0, 0.0, 500.0);
    auto t_ptc = timer::get_duration_since_stamp("us");
    Logger::print_info("Ptc Update took {}us", t_ptc);

    if (step % 50 == 0) {
      timer::stamp();
      data.particles.sort_by_cell();
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
