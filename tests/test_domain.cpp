#include "catch.hpp"
#include "cuda/core/cu_sim_data.h"
#include "cuda/core/sim_environment_dev.h"
#include "utils/util_functions.h"
#include <cstring>

using namespace Aperture;
using namespace std;

TEST_CASE("Making domain", "[Domain]") {
  int my_argc = 3;
  char** my_argv = new char*[my_argc + 1];
  for (int i = 0; i < my_argc; i++) my_argv[i] = new char[100];
  strcpy(my_argv[0], "aperture");
  strcpy(my_argv[1], "-c");
  strcpy(my_argv[2], "test_domain.toml");
  my_argv[3] = nullptr;

  cu_sim_environment env(&my_argc, &my_argv);

  cu_sim_data data(env);

  Logger::print_info("ptr size is {}, {}, {}",
                     data.E[0].grid().extent().width(),
                     data.E[0].grid().extent().height(),
                     data.E[0].grid().extent().depth());

  // REQUIRE(data.dev_map.size() == 2);
  if (data.dev_map.size() == 2) {
    REQUIRE(data.particles.size() == 2);
    REQUIRE(data.E.size() == 2);

    data.E[0].assign(1.0);
    data.E[1].assign(2.0);

    env.get_sub_guard_cells(data.E);
    data.E[0].sync_to_host();
    data.E[1].sync_to_host();

    auto& mesh = data.E[0].grid().mesh();
    for (int i = 0; i < mesh.dims[0]; i++) {
      REQUIRE(data.E[0](0, i, mesh.dims[1] - 1) == 2.0);
      REQUIRE(data.E[0](0, i, mesh.dims[1] - 2) == 2.0);
      REQUIRE(data.E[0](1, i, mesh.dims[1] - 1) == 2.0);
      REQUIRE(data.E[0](1, i, mesh.dims[1] - 2) == 2.0);
      REQUIRE(data.E[0](2, i, mesh.dims[1] - 1) == 2.0);
      REQUIRE(data.E[0](2, i, mesh.dims[1] - 2) == 2.0);
      REQUIRE(data.E[1](0, i, 0) == 1.0);
      REQUIRE(data.E[1](0, i, 1) == 1.0);
      REQUIRE(data.E[1](1, i, 0) == 1.0);
      REQUIRE(data.E[1](1, i, 1) == 1.0);
      REQUIRE(data.E[1](2, i, 0) == 1.0);
      REQUIRE(data.E[1](2, i, 1) == 1.0);
    }

    Logger::print_info("Getting field guard cells passed");

    data.J[0].assign(0.1);
    data.J[1].assign(0.2);
    data.Rho[0][0].assign(0.4);
    data.Rho[0][1].assign(0.2);
    env.send_sub_guard_cells(data.J);
    env.send_sub_guard_cells(data.Rho[0]);
    data.J[0].sync_to_host();
    data.J[1].sync_to_host();
    data.Rho[0][0].sync_to_host();
    data.Rho[0][1].sync_to_host();
    for (int i = 0; i < mesh.dims[0]; i++) {
      REQUIRE(data.J[0](0, i, mesh.dims[1] - 3) == 0.3f);
      REQUIRE(data.J[0](0, i, mesh.dims[1] - 4) == 0.3f);
      REQUIRE(data.J[0](1, i, mesh.dims[1] - 3) == 0.3f);
      REQUIRE(data.J[0](1, i, mesh.dims[1] - 4) == 0.3f);
      REQUIRE(data.J[0](2, i, mesh.dims[1] - 3) == 0.3f);
      REQUIRE(data.J[0](2, i, mesh.dims[1] - 4) == 0.3f);
      REQUIRE(data.J[1](0, i, 2) == 0.3f);
      REQUIRE(data.J[1](0, i, 3) == 0.3f);
      REQUIRE(data.J[1](1, i, 2) == 0.3f);
      REQUIRE(data.J[1](1, i, 3) == 0.3f);
      REQUIRE(data.J[1](2, i, 2) == 0.3f);
      REQUIRE(data.J[1](2, i, 3) == 0.3f);
      REQUIRE(data.Rho[0][0](i, mesh.dims[1] - 3) == 0.6f);
      REQUIRE(data.Rho[0][0](i, mesh.dims[1] - 4) == 0.6f);
      REQUIRE(data.Rho[0][1](i, 2) == 0.6f);
      REQUIRE(data.Rho[0][1](i, 3) == 0.6f);
    }

    Logger::print_info("Sending j and rho guard cells passed");

    Logger::print_debug("particles[0] initially has {} particles",
                        data.particles[0].number());
    // cudaSetDevice(data.dev_map[0]);
    data.particles[0].append({0.1f, 0.2f, 0.3f}, {0.0f, 0.0f, 0.0f},
                             mesh.get_idx(132, 66),
                             ParticleType::electron);
    data.particles[0].append({0.1f, 0.2f, 0.3f}, {0.0f, 0.0f, 0.0f},
                             mesh.get_idx(148, 32),
                             ParticleType::electron);
    Logger::print_debug("particles[0] now has {} particles",
                        data.particles[0].number());
    Logger::print_debug("particles[1] initially has {} particles",
                        data.particles[1].number());
    // cudaSetDevice(data.dev_map[1]);
    data.particles[1].append({0.32f, 0.15f, 0.26f}, {0.0f, 0.0f, 0.0f},
                             mesh.get_idx(99, 1),
                             ParticleType::positron);
    data.particles[1].append({0.32f, 0.15f, 0.26f}, {0.0f, 0.0f, 0.0f},
                             mesh.get_idx(229, 12),
                             ParticleType::positron);
    Logger::print_debug("particles[1] now has {} particles",
                        data.particles[1].number());
    data.send_particles();

    Logger::print_debug(
        "x1s: {}, {}, {}", data.particles[0].data().x1[0],
        data.particles[0].data().x1[1], data.particles[0].data().x1[2]);
    Logger::print_debug("cells: {}, {}, {}",
                        data.particles[0].data().cell[0],
                        data.particles[0].data().cell[1],
                        data.particles[0].data().cell[2]);
    REQUIRE(data.particles[0].data().x1[2] == 0.32f);
    REQUIRE(data.particles[0].data().cell[2] == mesh.get_idx(99, 65));
    REQUIRE(get_ptc_type(data.particles[0].data().flag[2]) == 1);
    REQUIRE(data.particles[1].data().x2[2] == 0.2f);
    REQUIRE(data.particles[1].data().cell[2] == mesh.get_idx(132, 2));
    REQUIRE(get_ptc_type(data.particles[1].data().flag[2]) == 0);

    Logger::print_info("Sending particles passed!");
  }
}
