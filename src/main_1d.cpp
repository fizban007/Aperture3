#include "algorithms/field_solver_1d.h"
#include "algorithms/ptc_updater_1d.h"
#include "sim_data.h"
#include "sim_environment.h"
#include "utils/data_exporter.h"
#include "utils/logger.h"
#include "utils/timer.h"
#include "utils/util_functions.h"
#include "radiation/rt_1d.h"
#include <random>

using namespace Aperture;

int
main(int argc, char* argv[]) {
  uint32_t step = 0;
  // Construct the simulation environment
  sim_environment env(&argc, &argv);

  // Allocate simulation data
  sim_data data(env);

  ptc_updater_1d ptcupdater(env);
  field_solver_1d solver;
  rad_transfer_1d rt(env);

  auto& grid = env.local_grid();
  auto& mesh = grid.mesh();

  // Setup initial conditions
  for (int i = mesh.guard[0]; i < mesh.dims[0] - mesh.guard[0]; i++) {
    data.particles.append(
        {env.gen_rand(), 0.0, 0.0}, {0.0, 0.0, 0.0}, i,
        ParticleType::electron, 1.0,
        (env.gen_rand() < 0.1 ? bit_or(ParticleFlag::tracked) : 0));
    data.particles.append(
        {env.gen_rand(), 0.0, 0.0}, {0.0, 0.0, 0.0}, i,
        ParticleType::positron, 1.0,
        (env.gen_rand() < 0.1 ? bit_or(ParticleFlag::tracked) : 0));
  }

  // Setup background J
  vector_field<Scalar> Jbg(grid);
  Jbg.initialize(0,
                 [](Scalar x1, Scalar x2, Scalar x3) { return 1.0; });

  data_exporter exporter(env.params(), step);

  exporter.add_field("E", data.E);
  exporter.add_field("J", data.J);
  exporter.add_field("Rho_e", data.Rho[0]);
  exporter.add_field("Rho_p", data.Rho[1]);
  exporter.add_ptc_output_1d("particles", "ptc", &data.particles);

  exporter.copyConfigFile();
  exporter.WriteGrid();

  for (; step <= env.params().max_steps; step++) {
    double dt = env.params().delta_t;
    double time = step * dt;

    if (step % env.params().data_interval == 0) {
      // Output data
      exporter.WriteOutput(step, time);
      exporter.write_particles(step, time);
    }

    ptcupdater.update_particles(data, dt, step);
    solver.update_fields(data.E, data.J, Jbg, dt, time);
    rt.emit_photons(data);
    rt.produce_pairs(data);

    if (step % 10 == 0) {
      data.photons.sort_by_cell(grid);
      data.particles.sort_by_cell(grid);
    }
    Logger::print_info("There are {} particles in the pool", data.particles.number());
    Logger::print_info("There are {} photons in the pool", data.photons.number());

    Logger::print_info("=== At timestep {}, time = {} ===", step, time);
  }

  return 0;
}
