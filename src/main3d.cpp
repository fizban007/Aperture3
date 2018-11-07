#include "algorithms/field_solver_default.h"
#include "ptc_updater.h"
#include "sim_data.h"
#include "sim_environment.h"
#include "utils/logger.h"
#include "utils/timer.h"
#include <random>

using namespace Aperture;

int
main(int argc, char* argv[]) {
  // Construct the simulation environment
  Environment env(&argc, &argv);

  // Allocate simulation data
  SimData data(env);

  // Initialize components of the simulator
  PtcUpdater ptc_updater(env);
  FieldSolver_Default field_solver(env.grid());

  // Apply initial conditions
  Scalar B0 = 100.0f;
  data.E.initialize();
  data.B.initialize();
  data.B.initialize(
      2, [B0](Scalar x1, Scalar x2, Scalar x3) { return B0; });

  std::default_random_engine gen;
  std::uniform_int_distribution<int> dist(10, 51);
  auto& mesh = env.grid().mesh();
  uint32_t N = 10000000;
  for (uint32_t i = 0; i < N; i++) {
    data.particles.append({0.f, 0.f, 0.f}, {5.0f, 0.0f, 1.0f},
                          mesh.get_idx(dist(gen), dist(gen), dist(gen)),
                          ParticleType::electron);
  }
  Logger::print_info("number of particles is {}",
                     data.particles.number());
  // data.particles.sync_to_device();

  // Update loop
  for (uint32_t step = 0; step < env.params().max_steps; step++) {
    double dt = env.params().delta_t;
    double time = step * dt;
    timer::stamp();
    ptc_updater.update_particles(data, dt);
    auto t_ptc = timer::get_duration_since_stamp("us");
    Logger::print_info("Ptc Update took {}us", t_ptc);

    if (step % 10 == 0) {
      timer::stamp();
      data.particles.sort_by_cell();
      auto t_sort = timer::get_duration_since_stamp("us");
      Logger::print_info("Ptc sort took {}us", t_sort);
    }

    timer::stamp();
    // ptc_updater.handle_boundary(data);
    field_solver.update_fields(data.E, data.B, data.J, dt, time);
    auto t_field = timer::get_duration_since_stamp("us");
    Logger::print_info("Field Update took {}us", t_field);
  }

  return 0;
}
