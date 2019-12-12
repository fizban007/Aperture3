#include "catch.hpp"
#include "core/particles.h"
#include <vector>

using namespace Aperture;

TEST_CASE("Copying to particle communication buffers, 3D",
          "[ParticleComm]") {
  int N1 = 12, N2 = 12, N3 = 12;
  Quadmesh mesh(N1, N2, N3);
  mesh.guard[0] = 2;
  mesh.guard[1] = 2;
  mesh.guard[2] = 2;

  particles_t ptc(10000);
  ptc.append({0.5, 0.5, 0.5}, {-1.0, 0.0, 0.0}, 1 + 5 * N1 + 6 * N1 * N2,
             ParticleType::electron);
  ptc.append({0.5, 0.5, 0.5}, {1.0, 0.0, 0.0}, 11 + 3 * N1 + 9 * N1 * N2,
             ParticleType::electron);
  ptc.append({0.5, 0.5, 0.5}, {-1.0, 1.0, 0.0}, 1 + 10 * N1 + 6 * N1 * N2,
             ParticleType::electron);
  ptc.append({0.5, 0.5, 0.5}, {1.0, -1.0, 0.0}, 11 + 0 * N1 + 8 * N1 * N2,
             ParticleType::electron);
  REQUIRE(ptc.number() == 4);

  std::vector<particles_t::base_class> buffers;
  for (int i = 0; i < 27; i++)
    buffers.emplace_back(100, true);

  ptc.copy_to_comm_buffers(buffers, mesh);
  // cudaDeviceSynchronize();

  for (int n = 0; n < 27; n++) {
    if (n <= 15 && n >= 11 && n != 13)
      REQUIRE(buffers[n].number() == 1);
    else
      REQUIRE(buffers[n].number() == 0);
  }
  REQUIRE(buffers[12].data().p1[0] == -1.0f);
  REQUIRE(buffers[14].data().p1[0] == 1.0f);
  REQUIRE(buffers[11].data().p2[0] == -1.0f);
  REQUIRE(buffers[15].data().p2[0] == 1.0f);
  // REQUIRE(buffers[15].data().p2[0] == Approx(0.0f));
}

TEST_CASE("Copying to particle communication buffers, 2D",
          "[ParticleComm]") {
  int N1 = 12, N2 = 12, N3 = 1;
  Quadmesh mesh(N1, N2, N3);
  mesh.guard[0] = 2;
  mesh.guard[1] = 2;
  mesh.guard[2] = 0;

  particles_t ptc(10000);
  ptc.append({0.5, 0.5, 0.5}, {-1.0, 0.0, 0.0}, 1 + 5 * N1,
             ParticleType::electron);
  ptc.append({0.5, 0.5, 0.5}, {1.0, 0.0, 0.0}, 11 + 3 * N1,
             ParticleType::electron);
  ptc.append({0.5, 0.5, 0.5}, {-1.0, 1.0, 0.0}, 1 + 10 * N1,
             ParticleType::electron);
  ptc.append({0.5, 0.5, 0.5}, {1.0, -1.0, 0.0}, 11 + 0 * N1,
             ParticleType::electron);

  std::vector<particles_t::base_class> buffers;
  for (int i = 0; i < 9; i++)
    buffers.emplace_back(100, true);

  ptc.copy_to_comm_buffers(buffers, mesh);
  // cudaDeviceSynchronize();

  for (int n = 0; n < 9; n++) {
    if (n <= 6 && n >= 2 && n != 4)
      REQUIRE(buffers[n].number() == 1);
    else
      REQUIRE(buffers[n].number() == 0);
  }
  REQUIRE(buffers[3].data().p1[0] == -1.0f);
  REQUIRE(buffers[5].data().p1[0] == 1.0f);
  REQUIRE(buffers[6].data().p2[0] == 1.0f);
  REQUIRE(buffers[2].data().p2[0] == -1.0f);
  // REQUIRE(buffers[6].data().p2[0] == Approx(0.0f));
}
