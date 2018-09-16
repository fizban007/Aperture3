#include "radiation/radiation_transfer.h"
#include "data/particles.h"
#include "data/photons.h"
#include "sim_environment.h"
#include "utils/timer.h"
#include "utils/logger.h"
#include "catch.hpp"

using namespace Aperture;

TEST_CASE("Producing photons", "[Photons]") {
  Environment env("test.toml");

  Particles ptc(env.params());
  Photons photons(env.params());

  RadiationTransfer<Particles, Photons> rad(env);

  ptc.append({0.5,0.5,0.0}, {10.0,0.0,0.0}, 128, ParticleType::electron);
}
