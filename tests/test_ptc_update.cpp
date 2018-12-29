#include "algorithms/ptc_updater_default.h"
#include "catch.hpp"
#include "data/fields.h"
#include "sim_data.h"
#include "sim_environment.h"
#include "utils/logger.h"
#include "utils/timer.h"

#include <random>

using namespace Aperture;

TEST_CASE("Particle Push", "[ptc_update]") {
  sim_environment env("test_ptc.toml");

  sim_data data(env);

  std::default_random_engine gen;
  std::uniform_int_distribution<uint32_t> dist(10, 260 - 10);
  std::uniform_real_distribution<float> dist_f(0.0, 1.0);

  auto& mesh = data.E.grid().mesh();

  const uint32_t N = 5000000;
  for (uint32_t i = 0; i < N; i++) {
    data.particles.append({dist_f(gen), dist_f(gen), dist_f(gen)},
                          {0.0, 0.0, 0.0},
                          mesh.get_idx(dist(gen), dist(gen), dist(gen)),
                          ParticleType::electron);
  }

  ptc_updater_default pusher(env);

  timer::stamp();

  pusher.update_particles(data, 0.001);
  auto t = timer::get_duration_since_stamp("us");
  Logger::print_info(
      "Boris push for {} particles took {}us.", N, t);
}
