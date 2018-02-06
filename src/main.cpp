#include <iostream>
#include "sim_environment.h"
#include "sim_data.h"
#include "pic_sim.h"

using namespace Aperture;

int main(int argc, char *argv[])
{
  // Initialize the simulation environment
  Environment env(&argc, &argv);

  // These are debug output
  std::cout << env.conf().delta_t << std::endl;
  std::cout << env.args().conf_filename() << std::endl;
  std::cout << env.args().data_interval() << std::endl;
  std::cout << env.args().steps() << std::endl;
  std::cout << env.args().dimx() << std::endl;

  // Allocate simulation data
  SimData data(env);

  // These are debug output
  // std::cout << data.particles.size() << std::endl;
  // std::cout << data.particles[0].numMax() << std::endl;
  // std::cout << data.particles[0].number() << std::endl;
  // std::cout << data.photons.numMax() << std::endl;
  // std::cout << data.photons.number() << std::endl;

  // std::cout << data.E.grid_size() << std::endl;
  // std::cout << data.B.extent() << std::endl;

  // Initialize data output
  env.exporter().AddArray("E1", data.E, 0);
  env.exporter().AddArray("J1", data.J, 0);
  env.exporter().AddArray("Rho_e", data.Rho[0].data());
  env.exporter().AddArray("Rho_p", data.Rho[1].data());

  PICSim sim(env);

  // Setup the initial condition of the simulation
  for (int i = 0; i < 1; i++) {
    data.particles[0].append(0.0, 1.0, 100);
    data.particles[1].append(0.0, -1.0, 100);
  }

  // Setup the background current
  auto &grid = data.E.grid();
  auto &mesh = grid.mesh();
  VectorField<Scalar> Jb(grid);
  for (int i = mesh.guard[0]; i < mesh.dims[0] - mesh.guard[0]; i++) {
    // x is the staggered position where current is evaluated
    Scalar x = mesh.pos(0, i, true);
    Jb(0, i) = 0.0;
  }
  sim.field_solver().set_background_j(Jb);

  // Some more debug output
  Logger::print_info("There are {} electrons in the initial setup", data.particles[0].number());
  Logger::print_info("There are {} positrons in the initial setup", data.particles[1].number());
  Logger::print_info("There are {} ions in the initial setup", data.particles[2].number());
  Logger::print_info("There are {} photons in the initial setup", data.photons.number());

  // Main simulation loop
  for (uint32_t step = 0; step < env.args().steps(); step++) {
    Logger::print_info("At time step {}", step);
    double time = step * env.conf().delta_t;

    sim.step(data, step);

    if (step % env.args().data_interval() == 0)
      env.exporter().WriteOutput(step, time);
  }
  return 0;
}
