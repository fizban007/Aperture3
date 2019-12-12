#include "catch.hpp"
#include "core/particles.h"
#include "cuda/constant_mem_func.h"
#include "utils/logger.h"
#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif
#include <vector>

using namespace Aperture;

TEST_CASE("Copying to particle communication buffers, 3D",
          "[ParticleComm]") {
  int N1 = 12, N2 = 12, N3 = 12;
  Quadmesh mesh(N1, N2, N3);
  mesh.guard[0] = 2;
  mesh.guard[1] = 2;
  mesh.guard[2] = 2;
  init_dev_mesh(mesh);

  particles_t ptc(10000);
  ptc.append({0.5, 0.5, 0.5}, {-1.0, 0.0, 0.0}, 1 + 5 * N1 + 6 * N1 * N2,
             ParticleType::electron);
  ptc.append({0.5, 0.5, 0.5}, {1.0, 0.0, 0.0}, 11 + 3 * N1 + 9 * N1 * N2,
             ParticleType::electron);
  ptc.append({0.5, 0.5, 0.5}, {-1.0, 1.0, 0.0}, 1 + 10 * N1 + 6 * N1 * N2,
             ParticleType::electron);
  ptc.append({0.5, 0.5, 0.5}, {1.0, -1.0, 0.0}, 11 + 0 * N1 + 8 * N1 * N2,
             ParticleType::electron);

  // cudaDeviceSynchronize();

  std::vector<particles_t::base_class> buffers;
  for (int n = 0; n < 27; n++) {
    buffers.push_back(std::move(particles_t::base_class(100, true)));
    // cudaDeviceSynchronize();
    // if (buffers.size() > 13)
    //   Logger::print_info("buffer {} has number {}", 13, buffers[13].number());
  }

  ptc.copy_to_comm_buffers(buffers, mesh);
  // cudaDeviceSynchronize();

  for (int n = 0; n < buffers.size(); n++) {
    // Logger::print_info("buffer {} has number {}", n, buffers[n].number());
    if (n <= 15 && n >= 11 && n != 13)
      REQUIRE(buffers[n].number() == 1);
    else
      REQUIRE(buffers[n].number() == 0);
  }
  REQUIRE(buffers[12].data().p1[0] == -1.0);
  REQUIRE(buffers[14].data().p1[0] == 1.0);
  REQUIRE(buffers[11].data().p2[0] == -1.0);
  REQUIRE(buffers[15].data().p2[0] == 1.0);
  REQUIRE(buffers[16].data().p2[0] == 0.0);
  REQUIRE(buffers[13].data().p2[0] == 0.0);
}

// TEST_CASE("Copying to particle communication buffers, 2D",
//           "[ParticleComm]") {
//   int N1 = 12, N2 = 12, N3 = 1;
//   Quadmesh mesh(N1, N2, N3);
//   mesh.guard[0] = 2;
//   mesh.guard[1] = 2;
//   mesh.guard[2] = 0;
//   init_dev_mesh(mesh);

//   particles_t ptc(10000);
//   ptc.append({0.5, 0.5, 0.5}, {-1.0, 0.0, 0.0}, 1 + 5 * N1,
//              ParticleType::electron);
//   ptc.append({0.5, 0.5, 0.5}, {1.0, 0.0, 0.0}, 11 + 3 * N1,
//              ParticleType::electron);
//   ptc.append({0.5, 0.5, 0.5}, {-1.0, 1.0, 0.0}, 1 + 10 * N1,
//              ParticleType::electron);
//   ptc.append({0.5, 0.5, 0.5}, {1.0, -1.0, 0.0}, 11 + 0 * N1,
//              ParticleType::electron);

//   std::vector<particles_t::base_class> buffers;
//   for (int i = 0; i < 9; i++)
//     buffers.emplace_back(100, true);

//   ptc.copy_to_comm_buffers(buffers, mesh);
//   // cudaDeviceSynchronize();

//   for (int n = 0; n < 9; n++) {
//     if (n <= 6 && n >= 2 && n != 4)
//       REQUIRE(buffers[n].number() == 1);
//     else
//       REQUIRE(buffers[n].number() == 0);
//   }
//   REQUIRE(buffers[3].data().p1[0] == -1.0);
//   REQUIRE(buffers[5].data().p1[0] == 1.0);
//   REQUIRE(buffers[6].data().p2[0] == 1.0);
//   REQUIRE(buffers[2].data().p2[0] == -1.0);
//   REQUIRE(buffers[7].data().p2[0] == 0.0);
// }
