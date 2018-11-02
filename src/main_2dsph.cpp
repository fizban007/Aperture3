#include "algorithms/field_solver_log_sph.h"
#include "ptc_updater_logsph.h"
#include "sim_data.h"
#include "sim_environment.h"
#include "utils/logger.h"
#include "utils/timer.h"
#include <random>

using namespace Aperture;

int main(int argc, char *argv[])
{
  // Construct the simulation environment
  Environment env(&argc, &argv);

  // Allocate simulation data
  SimData data(env);

  // Initialize the field solver
  FieldSolver_LogSph field_solver(*dynamic_cast<const Grid_LogSph*>(&env.local_grid()));

  // Initialize particle updater
  PtcUpdaterLogSph ptc_updater(env);

  // Initialize data exporter
  DataExporter exporter(env.params(), "/home/alex/storage/Data/Aperture3/2d_test/",
                        "data", 1);
  exporter.WriteGrid();

  Scalar B0 = 1000.0;
  data.E.initialize();
  data.B.initialize();
  data.B.initialize(0, [B0](Scalar x1, Scalar x2, Scalar x3) {
                         Scalar r = exp(x1);
                         return 2.0 * B0 * cos(x2) / (r * r * r);
                       });
  data.B.initialize(1, [B0](Scalar x1, Scalar x2, Scalar x3) {
                         Scalar r = exp(x1);
                         return B0 * sin(x2) / (r * r * r);
                       });
  data.B.sync_to_device();
  exporter.AddField("E", data.E);
  exporter.AddField("B", data.B);
  exporter.AddField("J", data.J);
  exporter.AddField("Rho_e", data.Rho[0]);

  // Initialize a bunch of particles
  std::default_random_engine gen;
  std::uniform_int_distribution<int> dist(100, 258);
  auto &mesh = env.grid().mesh();
  uint32_t N = 1;
  for (uint32_t i = 0; i < N; i++) {
    data.particles.append({0.f, 0.f, 0.f}, {0.0f, -5.0f, 0.0f},
                          // mesh.get_idx(dist(gen), dist(gen)),
                          mesh.get_idx(100, 258),
                          ParticleType::electron, 1000.0);
  }
  Logger::print_info("number of particles is {}",
                     data.particles.number());
  data.particles.sync_to_device();

  // exporter.WriteOutput(0, 0.0);
  // exporter.writeXMF(0, 0.0);

  for (uint32_t step = 0; step < env.params().max_steps; step++) {
    double dt = env.params().delta_t;
    // double dt = 0.0;
    double time = step * dt;
    Logger::print_info("At timestep {}, time = {}", step, time);

    if ((step % env.params().data_interval) == 0) {
      data.E.sync_to_host();
      data.B.sync_to_host();
      data.J.sync_to_host();
      data.Rho[0].sync_to_host();

      exporter.WriteOutput(step, time);
      exporter.writeXMF(step, time);
    }
    
    timer::stamp();
    ptc_updater.update_particles(data, dt);
    auto t_ptc = timer::get_duration_since_stamp("us");
    Logger::print_info("Ptc Update took {}us", t_ptc);

    if (step % 50 == 0) {
      timer::stamp();
      data.particles.sort_by_cell();
      auto t_sort = timer::get_duration_since_stamp("us");
      Logger::print_info("Ptc sort took {}us", t_sort);
    }

    // timer::stamp();
    // // ptc_updater.handle_boundary(data);
    // field_solver.update_fields(data.E, data.B, data.J, dt, time);
    // auto t_field = timer::get_duration_since_stamp("us");
    // Logger::print_info("Field Update took {}us", t_field);
  }
  return 0;
}
