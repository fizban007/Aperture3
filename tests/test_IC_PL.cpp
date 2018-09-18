#include <random>
#include "radiation/inverse_compton_power_law.h"
#include "catch.hpp"

using namespace Aperture;

struct Rng
{
  std::default_random_engine gen;
  std::uniform_real_distribution<float> dist;

  Rng() : dist(0.0f, 1.0f) {}
  ~Rng() {}

  float operator()() {
    return dist(gen);
  }
};

TEST_CASE("Simple rng generation", "[inverse_compton]") {
  Rng rng;

  auto IC = make_inverse_compton_PL1D(SimParams{}, rng);
  IC.draw_photon_energy(1000.0, 999.0);
}
