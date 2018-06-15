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
    ptc.append(0.1, 0.0, i, ParticleType::electron);
  }
  ptc.set_num(N);

  ptc.compute_tile_num();

  ptc.sync_to_device(0);

  timer::stamp();
  // ptc.sort_by_cell();
  ptc.sort_by_tile(tile_size);
  timer::show_duration_since_stamp("sorting by tile on gpu, with copies", "ms");
}
