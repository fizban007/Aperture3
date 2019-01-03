#include "pic_sim.h"
#include "cu_sim_data.h"
#include "sim_environment_dev.h"
#include "utils/logger.h"
#include "utils/util_functions.h"
#include <iostream>
#include <random>

using namespace Aperture;

// Define background charge density profile here
double
rho_gj(double jb, double x) {
  // return jb * (0.85 - 130.0 / (80.0 + 250.0 * (x - 0.03)));
  // return jb * 1.75* std::atan(1.0 * (2.0 * x - 1.2)) * 2.0 /
  // CONST_PI; return jb * (0.44 + 0.6 * std::atan(5.0 * (2.0 * x
  // - 1.2)) * 2.0 / CONST_PI); return jb * 0.9 * (2.0 * x - 1.0);
  return jb * 0.7;
}

int
main(int argc, char *argv[]) {
  // Initialize the simulation environment
  Environment env(&argc, &argv);

  // These are debug output
  Logger::print_debug("{}", env.params().delta_t);
  Logger::print_debug("{}", env.params().conf_file);
  Logger::print_debug("{}", env.params().data_interval);
  Logger::print_debug("{}", env.params().max_steps);

  // Allocate simulation data
  cu_sim_data data(env);

  // These are debug output
  // std::cout << data.particles.size() << std::endl;
  // std::cout << data.particles[0].size() << std::endl;
  // std::cout << data.particles[0].number() << std::endl;
  // std::cout << data.photons.size() << std::endl;
  // std::cout << data.photons.number() << std::endl;

  // std::cout << data.E.grid_size() << std::endl;
  // std::cout << data.B.extent() << std::endl;

  PICSim sim(env);
  auto &grid = data.E.grid();
  auto &mesh = grid.mesh();
  // int ppc = env.params().ptc_per_cell;

  // Setup the initial condition of the simulation
  // for (int i = mesh.guard[0]; i < mesh.dims[0] - mesh.guard[0]; i++)
  // { data.particles[1].append(env.gen_rand(), 0.0, i,
  //                          (env.gen_rand() <
  //                          env.params().track_percent ?
  //                          (int)ParticleFlag::tracked : 0));
  //   for (int n = 0; n < ppc; n++) {
  //     data.particles[1].append(env.gen_rand(), 100.0 - 10.0
  //     + 20.0*env.gen_rand(), i,
  //                              (env.gen_rand() <
  //                              env.params().track_percent ?
  //                              (int)ParticleFlag::tracked : 0));
  //     // data.particles[0].append(dist(generator), 0.99 + 0.02 *
  //     dist(generator), i, data.particles[0].append(env.gen_rand(),
  //     -100.0
  //     + 10.0 + 20.0*env.gen_rand(), i,
  //                              (env.gen_rand() <
  //                              env.params().track_percent ?
  //                              (int)ParticleFlag::tracked : 0));
  //   }
  // }

  // for (int i = mesh.dims[0] / 4; i < mesh.dims[0] * 3/ 4; i++) {
  //   data.particles[0].append(env.gen_rand(), 0.0, i,
  //                            (env.gen_rand() <
  //                            env.params().track_percent ?
  //                            (int)ParticleFlag::tracked : 0));
  //   data.particles[1].append(env.gen_rand(), 0.0, i,
  //                            (env.gen_rand() <
  //                            env.params().track_percent ?
  //                            (int)ParticleFlag::tracked : 0));
  // }

  // for (int i = mesh.dims[0] * 3 / 4; i < mesh.dims[0]-mesh.guard[0];
  // i++) {
  //   data.particles[0].append(env.gen_rand(), 0.0, i,
  //                            (env.gen_rand() <
  //                            env.params().track_percent ?
  //                            (int)ParticleFlag::tracked : 0));
  //   for (int n = 0; n < ppc; n++) {
  //     // data.particles[0].append(dist(generator), 0.99 + 0.02 *
  //     dist(generator), i,
  //     data.particles[1].append(env.gen_rand(), 1.0, i,
  //                              (env.gen_rand() <
  //                              env.params().track_percent ?
  //                              (int)ParticleFlag::tracked : 0));
  //   }
  // }
  double jb = 1.0;
  double initial_M = 20.0;
  for (int i = mesh.guard[0]; i < mesh.dims[0] - mesh.guard[0]; i++) {
    // double rho = -jb * 0.5 * (2.0 * i / (double)mesh.reduced_dim(0)
    // - 1.3) / env.params().q_e; double rho = jb * (0.85 - 130.0 /
    // (80.0 + 250.0 * i / (double)mesh.reduced_dim(0))) /
    // env.params().q_e; double rho = jb * 0.5 * (1.85 - 130.0 / (80.0 +
    // 250.0 * i / (double)mesh.reduced_dim(0))) / env.params().q_e;
    double rho =
        rho_gj(jb, i / (double)mesh.reduced_dim(0)) / env.params().q_e;
    // double rho = (double)ppc * 0.2 * cos(2.0 * acos(-1.0) * i /
    // (double)mesh.reduced_dim(0)); double rho = 0.0; if (i < 0.5 *
    // (mesh.guard[0] + mesh.reduced_dim(0)))
    //   rho = (double)ppc * 0.2 * (1.0 - 1.8 * i /
    //   (double)mesh.reduced_dim(0));
    // else
    //   rho = (double)ppc * 0.2 * (1.8 * i /
    //   (double)mesh.reduced_dim(0) - 0.8);
    // for (int n = 0; n < 0.5*((jb *
    // initial_M)/env.params().q_e-std::abs(rho)); n++) { for (int n =
    // 0; n < 0.5*((jb * initial_M)/env.params().q_e); n++) {
    // data.particles.append(env.gen_rand(), 0.0 * sgn(2.0 * i /
    // mesh.reduced_dim(0) - 1.3), i,
    //                       ParticleType::electron, 1.0,
    //                       (env.gen_rand() <
    //                       env.params().track_percent ?
    //                       (int)ParticleFlag::tracked : 0));
    // data.particles.append(env.gen_rand(), 0.0 * sgn(2.0 * i /
    // mesh.reduced_dim(0) - 1.3), i,
    //                       ParticleType::positron, 1.0,
    //                       (env.gen_rand() <
    //                       env.params().track_percent ?
    //                       (int)ParticleFlag::tracked : 0));
    // data.particles[1].append(env.gen_rand(), 0.0 * sgn(2.0 * i /
    // mesh.reduced_dim(0) - 1.3), i,
    //                          (env.gen_rand() <
    //                          env.params().track_percent ?
    //                          (int)ParticleFlag::tracked : 0));
    // }

    // for (int n = 0; n < std::abs(rho); n++) {
    //   if (rho < 0)
    //     data.particles[0].append(env.gen_rand(), 0.0, i,
    //                              (env.gen_rand() <
    //                              env.params().track_percent ?
    //                              (int)ParticleFlag::tracked : 0));
    //   else
    //     data.particles[1].append(env.gen_rand(), 0.0, i,
    //                              (env.gen_rand() <
    //                              env.params().track_percent ?
    //                              (int)ParticleFlag::tracked : 0));
    // }
    for (int n = 0; n < jb * initial_M; n++) {
      data.particles.append(env.gen_rand(), 1.0, i,
                            ParticleType::electron, 1.0,
                            (env.gen_rand() < env.params().track_percent
                                 ? (int)ParticleFlag::tracked
                                 : 0));
      data.particles.append(env.gen_rand(), -1.0, i,
                            ParticleType::positron, 1.0,
                            (env.gen_rand() < env.params().track_percent
                                 ? (int)ParticleFlag::tracked
                                 : 0));
    }
  }

  // Setup the background current
  cu_vector_field<Scalar> Jb(grid);
  for (int i = mesh.guard[0] - 1; i < mesh.dims[0] - mesh.guard[0];
       i++) {
    // x is the staggered position where current is evaluated
    // Scalar x = mesh.pos(0, i, true);
    // Jb(0, i) = 1.0 + 9.0 * sin(CONST_PI * x / mesh.sizes[0]);
    Jb(0, i) = jb;
  }
  sim.field_solver().set_background_j(Jb);

  // Setup initial electric field
  // for (int i = mesh.guard[0] - 1; i < mesh.dims[0] - mesh.guard[0];
  // i++) {
  //   double x = mesh.pos(0, i, 1);
  //   data.E(0, i) = -jb * 1 * (19182.2 + 0.85 * x - 2600.0 * log(x +
  //   1600.0));
  // }

  // Initialize data output
  env.exporter().AddArray("E1", data.E, 0);
  // env.exporter().AddArray("E1avg", data.B, 0);
  env.exporter().AddArray("J1", data.J, 0);
  env.exporter().AddArray("Rho_e",
                          data.Rho[(int)ParticleType::electron].data());
  env.exporter().AddArray("Rho_p",
                          data.Rho[(int)ParticleType::positron].data());
  // env.exporter().AddArray("Rho_e_avg", data.Rho_avg[0].data());
  // env.exporter().AddArray("Rho_p_avg", data.Rho_avg[1].data());
  // env.exporter().AddArray("J_e_avg", data.J_avg[0].data());
  // env.exporter().AddArray("J_p_avg", data.J_avg[1].data());
  env.exporter().AddParticleArray("Electrons", data.particles);
  // env.exporter().AddParticleArray("Positrons", data.particles[1]);
  // if (env.params().trace_photons)
  //   env.exporter().AddParticleArray("Photons", data.photons);
  env.exporter().writeConfig(env.params());

  // Some more debug output
  Logger::print_info("There are {} particles in the initial setup",
                     data.particles.number());
  // Logger::print_info("There are {} positrons in the initial setup",
  // data.particles[1].number()); Logger::print_info("There are {} ions
  // in the initial setup", data.particles[2].number());
  // Logger::print_info("There are
  // {} photons in the initial setup", data.photons.number());

  // Main simulation loop
  for (uint32_t step = 0; step < env.params().max_steps; step++) {
    Logger::print_info("At time step {}", step);
    double time = step * env.params().delta_t;

    if (step % env.params().data_interval == 0) {
      //   double factor = 1.0 / env.args().data_interval();
      //   data.B.multiplyBy(factor);
      //   data.Rho_avg[0].multiplyBy(factor);
      //   data.Rho_avg[1].multiplyBy(factor);
      //   data.J_avg[0].multiplyBy(factor);
      //   data.J_avg[1].multiplyBy(factor);
      env.exporter().WriteOutput(step, time);
      //   data.B.initialize();
      //   data.Rho_avg[0].initialize();
      //   data.Rho_avg[1].initialize();
      //   data.J_avg[0].initialize();
      //   data.J_avg[1].initialize();
    }

    sim.step(data, step);
    // data.Rho_avg[0].addBy(data.Rho[0]);
    // data.Rho_avg[1].addBy(data.Rho[1]);
    // data.J_avg[0].addBy(data.J_s[0]);
    // data.J_avg[1].addBy(data.J_s[1]);
  }
  return 0;
}
