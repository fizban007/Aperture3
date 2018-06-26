#include "data/particles.h"
#include "utils/timer.h"
#include "utils/logger.h"
#include <random>
#include "sim_environment.h"
#include "catch.hpp"

using namespace Aperture;

TEST_CASE("Sorting Particles by tile", "[Particles]") {
  Environment env("test.toml");
  size_t N = 10000000;
  Particles ptc(N);

  for (size_t i = 0; i < N; i++) {
    ptc.append({1.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, i, ParticleType::electron);
  }
  ptc.set_num(N);

  ptc.compute_tile_num();

  ptc.sync_to_device(0);

  timer::stamp();
  // ptc.sort_by_cell();
  ptc.sort_by_tile();
  timer::show_duration_since_stamp("sorting by tile on gpu, with copies", "ms");
}

TEST_CASE("Erasing particles in guard cells", "[Particles]") {
  Environment env("test.toml");
  auto& mesh = env.mesh();
  size_t N = 10000000;
  Particles ptc(N);

  ptc.append({0.1, 0.2, 0.1}, {0.1, 0.2, 0.1}, mesh.get_idx(1, 0, 0), ParticleType::electron);
  ptc.append({0.1, 0.2, 0.1}, {0.1, 0.2, 0.1}, mesh.get_idx(100, 1, 2), ParticleType::positron);
  ptc.append({0.1, 0.2, 0.1}, {0.1, 0.2, 0.1}, mesh.get_idx(10004, 1, 2), ParticleType::electron);
  ptc.append({0.1, 0.2, 0.1}, {0.1, 0.2, 0.1}, mesh.get_idx(100, 1, 10), ParticleType::electron);

  CHECK(ptc.number() == 4);

  ptc.clear_guard_cells();
  CHECK(ptc.data().cell[0] == MAX_CELL);
  CHECK(ptc.data().cell[1] == mesh.get_idx(100, 1, 2));
  CHECK(ptc.data().cell[2] == MAX_CELL);
  CHECK(ptc.data().cell[3] == MAX_CELL);

  // Checking if MAX_CELL gives MAX_TILE as well
  ptc.compute_tile_num();
  CHECK(ptc.data().tile[0] == MAX_TILE);
  CHECK(ptc.data().tile[1] == 1);
  CHECK(ptc.data().tile[2] == MAX_TILE);
  CHECK(ptc.data().tile[3] == MAX_TILE);

  // Sort the particle now to see if the number updates
  ptc.sort_by_cell();
  CHECK(ptc.number() == 1);
}
