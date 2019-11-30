#include "catch.hpp"
#include "core/grid.h"
#include "cuda/data/particles_1d.h"
// #include "cu_sim_data.h"
#include "cuda/core/cu_sim_environment.h"
#include "utils/logger.h"
#include "utils/timer.h"
#include <random>
#include <vector>

using namespace Aperture;

TEST_CASE("Sorting Particles by cell", "[Particles]") {
  // cu_sim_environment env("test.toml");
  size_t N = 1000;
  Particles_1D ptc(N);

  for (size_t i = 0; i < N; i++) {
    ptc.append(1.0, 0.0, N - i, ParticleType::electron);
  }
  ptc.set_num(N);

  // ptc.compute_tile_num();

  // ptc.copy_to_device(0);

  timer::stamp();
  ptc.sort_by_cell();
  timer::show_duration_since_stamp(
      "sorting by cell on gpu, with copies", "ms");
}

// TEST_CASE("Sorting particles on CPU", "[Particles]") {
//   int N1 = 106, N2 = 5;
//   Grid grid(N1, N2);
//   auto& mesh = grid.mesh();
//   mesh.guard[0] = 3;
//   mesh.guard[1] = 2;
//   mesh.guard[2] = 0;

//   size_t N = 100000;
//   particles_t ptc(N);
//   // auto cell_ptr = thrust::device_pointer_cast(ptc.data().cell);

//   ptc.append({0.11, 0.12, 0.13}, {0.14, 0.15, 0.16},
//              mesh.get_idx(1, 0, 0), ParticleType::electron, 1.0);
//   ptc.append({0.21, 0.22, 0.23}, {0.24, 0.25, 0.26},
//              mesh.get_idx(100, 2, 0), ParticleType::positron, 2.0);
//   ptc.append({0.31, 0.32, 0.33}, {0.34, 0.35, 0.36},
//              mesh.get_idx(10004, 1, 0), ParticleType::electron, 3.0);
//   ptc.append({0.41, 0.42, 0.43}, {0.44, 0.45, 0.46},
//              mesh.get_idx(100, 1, 0), ParticleType::electron, 4.0);

//   CHECK(ptc.number() == 4);

//   // CHECK(ptc.data().cell[0] == MAX_CELL);
//   // CHECK(ptc.data().cell[1] == mesh.get_idx(100, 2, 2));
//   // CHECK(ptc.data().cell[2] == MAX_CELL);
//   // CHECK(ptc.data().cell[3] == MAX_CELL);

//   // Checking if MAX_CELL gives MAX_TILE as well
//   // ptc.compute_tile_num();
//   // CHECK(ptc.data().tile[0] == MAX_TILE);
//   // CHECK(ptc.data().tile[1] == 1);
//   // CHECK(ptc.data().tile[2] == MAX_TILE);
//   // CHECK(ptc.data().tile[3] == MAX_TILE);

//   // Sort the particle now to see if the number updates
//   ptc.sort_by_cell(grid);

//   CHECK(ptc.data().cell[0] == mesh.get_idx(100, 2));
//   CHECK(ptc.data().x1[0] == Approx(0.21f));
//   CHECK(ptc.number() == 4);

// }

// TEST_CASE("Sorting random particles on CPU", "[Particles]") {
//   int N1 = 400, N2 = 400;
//   Grid grid(N1, N2);
//   auto& mesh = grid.mesh();
//   mesh.guard[0] = 3;
//   mesh.guard[1] = 2;
//   mesh.guard[2] = 0;
//   size_t N = 6000000;
//   particles_t ptc(N);

//   std::default_random_engine gen;
//   std::uniform_int_distribution<int> dist(200, 300);
//   std::uniform_real_distribution<float> dist_f(0.0, 1.0);
//   uint32_t num = 5000000;

//   for (uint32_t i = 0; i < num; i++) {
//     ptc.append(
//         {dist_f(gen), dist_f(gen), 0.0}, {0.0, 0.0, 0.0},
//         (i % 10 == 0 ? MAX_CELL : mesh.get_idx(dist(gen), dist(gen))),
//         ParticleType::electron, 1.0);
//   }
//   // for (int i = 0; i < 100; i++) {
//   //   ptc.append({0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, MAX_CELL,
//   //              ParticleType::electron);
//   // }
//   // for (int i = 0; i < 900; i++) {
//   //   ptc.append({0.0, 0.0, 0.0}, {0.0, 0.0, 0.0},
//   //              mesh.get_idx(1, dist(gen)), ParticleType::electron);
//   // }
//   CHECK(ptc.number() == num);
//   // ptc.clear_guard_cells();
//   timer::stamp();
//   ptc.sort_by_cell(grid);

//   CHECK(ptc.number() == num * 9 / 10);
//   timer::show_duration_since_stamp(
//       "sorting 5M particles by cell on cpu", "ms");
// }

// TEST_CASE("Making photons", "[Particles]") {
//   cu_sim_environment env("test_particles.toml");
//   auto& mesh = env.grid().mesh();
//   cu_sim_data data(env);
//   RadiationTransferPulsar rad(env);

//   for (int i = 0; i < 1000; i++) {
//     data.particles.append({0.0, 0.0, 0.0},
//                           {(i % 10 == 0 ? 1000.0f : 0.0f), 0.0, 0.0},
//                           mesh.get_idx(10, 10),
//                           ParticleType::electron);
//   }
//   rad.emit_photons(data);
//   data.photons.sort_by_cell();
//   CHECK(data.photons.number() == 100);
//   rad.produce_pairs(data);
//   data.particles.sort_by_cell();
//   CHECK(data.particles.number() == 1200);
//   data.photons.sort_by_cell();
//   CHECK(data.photons.number() == 0);
// }

// TEST_CASE("Fine testing pair creation", "[Particles]") {
//   cu_sim_environment env("test_particles.toml");
//   auto& mesh = env.grid().mesh();
//   cu_sim_data data(env);
//   RadiationTransferPulsar rad(env);

//   std::vector<uint32_t> cell_ptr(10);
//   std::vector<Pos_t> x1_ptr(10);
//   std::vector<Pos_t> x2_ptr(10);
//   std::vector<Pos_t> x3_ptr(10);
//   std::vector<Scalar> weight_ptr(10);

//   data.particles.append({0.2, 0.3, 0.4},
//                         {1000.0, 0.0, 0.0},
//                         mesh.get_idx(10, 10),
//                         ParticleType::electron);
//   rad.emit_photons(data);

//   cudaMemcpy(x1_ptr.data(), data.photons.data().x1, 4 *
//   sizeof(Pos_t),
//              cudaMemcpyDeviceToHost);
//   cudaMemcpy(x2_ptr.data(), data.photons.data().x2, 4 *
//   sizeof(Pos_t),
//              cudaMemcpyDeviceToHost);
//   cudaMemcpy(x3_ptr.data(), data.photons.data().x3, 4 *
//   sizeof(Pos_t),
//              cudaMemcpyDeviceToHost);

//   for (int i = 0; i < 4; i++) {
//     Logger::print_info("photon, cell is {}, x is ({}, {}, {}), weight
//     is {}", cell_ptr[i], x1_ptr[i], x2_ptr[i], x3_ptr[i],
//     weight_ptr[i]);
//   }

//   rad.produce_pairs(data);
//   data.particles.sort_by_cell();
//   CHECK(data.particles.number() == 3);
//   data.photons.sort_by_cell();
//   CHECK(data.photons.number() == 0);

//   cudaMemcpy(cell_ptr.data(), data.particles.data().cell, 4 *
//   sizeof(uint32_t),
//              cudaMemcpyDeviceToHost);
//   cudaMemcpy(x1_ptr.data(), data.particles.data().x1, 4 *
//   sizeof(Pos_t),
//              cudaMemcpyDeviceToHost);
//   cudaMemcpy(x2_ptr.data(), data.particles.data().x2, 4 *
//   sizeof(Pos_t),
//              cudaMemcpyDeviceToHost);
//   cudaMemcpy(x3_ptr.data(), data.particles.data().x3, 4 *
//   sizeof(Pos_t),
//              cudaMemcpyDeviceToHost);
//   cudaMemcpy(weight_ptr.data(), data.particles.data().weight, 4 *
//   sizeof(Scalar),
//              cudaMemcpyDeviceToHost);
//   for (int i = 0; i < 4; i++) {
//     Logger::print_info("cell is {}, x is ({}, {}, {}), weight is {}",
//     cell_ptr[i], x1_ptr[i], x2_ptr[i], x3_ptr[i], weight_ptr[i]);
//   }
// }
