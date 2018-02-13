#include <iostream>
#include <random>
#include "sim_environment.h"
#include "sim_data.h"
#include "pic_sim.h"
#include "utils/util_functions.h"

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

  PICSim sim(env);
  auto &grid = data.E.grid();
  auto &mesh = grid.mesh();
  int ppc = 5;

  // Setup the initial condition of the simulation
  // for (int i = mesh.guard[0]; i < mesh.dims[0] / 4; i++) {
  //   data.particles[1].append(env.gen_rand(), 0.0, i,
  //                            (env.gen_rand() < env.conf().track_percent ? (int)ParticleFlag::tracked : 0));
  //   for (int n = 0; n < ppc; n++) {
  //     // data.particles[0].append(dist(generator), 0.99 + 0.02 * dist(generator), i,
  //     data.particles[0].append(env.gen_rand(), -1.0, i,
  //                              (env.gen_rand() < env.conf().track_percent ? (int)ParticleFlag::tracked : 0));
  //   }
  // }


  // for (int i = mesh.dims[0] / 4; i < mesh.dims[0] * 3/ 4; i++) {
  //   data.particles[0].append(env.gen_rand(), 0.0, i,
  //                            (env.gen_rand() < env.conf().track_percent ? (int)ParticleFlag::tracked : 0));
  //   data.particles[1].append(env.gen_rand(), 0.0, i,
  //                            (env.gen_rand() < env.conf().track_percent ? (int)ParticleFlag::tracked : 0));
  // }

  // for (int i = mesh.dims[0] * 3 / 4; i < mesh.dims[0]-mesh.guard[0]; i++) {
  //   data.particles[0].append(env.gen_rand(), 0.0, i,
  //                            (env.gen_rand() < env.conf().track_percent ? (int)ParticleFlag::tracked : 0));
  //   for (int n = 0; n < ppc; n++) {
  //     // data.particles[0].append(dist(generator), 0.99 + 0.02 * dist(generator), i,
  //     data.particles[1].append(env.gen_rand(), 1.0, i,
  //                              (env.gen_rand() < env.conf().track_percent ? (int)ParticleFlag::tracked : 0));
  //   }
  // }
  for (int i = mesh.guard[0]; i < mesh.dims[0] - mesh.guard[0]; i++) {
    double rho = (double)ppc * (2.0 * i / (double)mesh.reduced_dim(0) - 1.0) + 1.0;
    for (int n = 0; n < ppc * 5; n++) {
      data.particles[0].append(env.gen_rand(), 1.0 * sgn(rho - 2.0), i,
                               (env.gen_rand() < env.conf().track_percent ? (int)ParticleFlag::tracked : 0));
      data.particles[1].append(env.gen_rand(), 1.0 * sgn(rho - 2.0), i,
                               (env.gen_rand() < env.conf().track_percent ? (int)ParticleFlag::tracked : 0));
    }

    for (int n = 0; n < std::abs(rho); n++) {
      if (rho < 0)
        data.particles[0].append(env.gen_rand(), 0.0, i,
                                 (env.gen_rand() < env.conf().track_percent ? (int)ParticleFlag::tracked : 0));
      else
        data.particles[1].append(env.gen_rand(), 0.0, i,
                                 (env.gen_rand() < env.conf().track_percent ? (int)ParticleFlag::tracked : 0));
    }
  }

  // Setup the background current
  VectorField<Scalar> Jb(grid);
  for (int i = mesh.guard[0] - 1; i < mesh.dims[0] - mesh.guard[0]; i++) {
    // x is the staggered position where current is evaluated
    Scalar x = mesh.pos(0, i, true);
    // Jb(0, i) = 1.0 + 9.0 * sin(CONST_PI * x / mesh.sizes[0]);
    Jb(0, i) = 10.0;
  }
  sim.field_solver().set_background_j(Jb);

  // Initialize data output
  env.exporter().AddArray("E1", data.E, 0);
  env.exporter().AddArray("J1", data.J, 0);
  env.exporter().AddArray("Rho_e", data.Rho[0].data());
  env.exporter().AddArray("Rho_p", data.Rho[1].data());
  env.exporter().AddParticleArray("Electrons", data.particles[0]);
  env.exporter().AddParticleArray("Positrons", data.particles[1]);
  if (env.conf().trace_photons)
    env.exporter().AddParticleArray("Photons", data.photons);
  env.exporter().setGrid(grid);
  env.exporter().writeConfig(env.conf_file());

  // Some more debug output
  Logger::print_info("There are {} electrons in the initial setup", data.particles[0].number());
  Logger::print_info("There are {} positrons in the initial setup", data.particles[1].number());
  Logger::print_info("There are {} ions in the initial setup", data.particles[2].number());
  Logger::print_info("There are {} photons in the initial setup", data.photons.number());

  // Main simulation loop
  for (uint32_t step = 0; step < env.args().steps(); step++) {
    Logger::print_info("At time step {}", step);
    double time = step * env.conf().delta_t;

    if (step % env.args().data_interval() == 0)
      env.exporter().WriteOutput(step, time);

    sim.step(data, step);
  }
  return 0;
}
