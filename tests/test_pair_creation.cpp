#include "radiation/radiation_transfer.h"
#include "radiation/inverse_compton_dummy.h"
#include "data/particles.h"
#include "data/photons.h"
#include "sim_environment.h"
#include "utils/timer.h"
#include "utils/logger.h"
#include "cuda/cudarng.h"
#include "catch.hpp"

using namespace Aperture;

TEST_CASE("Producing photons", "[Photons]") {
  Environment env("test.toml");

  Particles ptc(env.params());
  Photons photons(env.params());

  RadiationTransfer<Particles, Photons, InverseComptonDummy<Kernels::CudaRng>> rad(env);

  int N = 10;
  for (int i = 0; i < N; i++) {
    float u = env.gen_rand() * 30;
    ptc.append({0.5,0.5,0.0}, {u,0.0,0.0}, 128, ParticleType::electron);
  }
  Logger::print_info("Current Particle momenta:");
  for (int j = 0; j < ptc.number(); j++) {
    Logger::print_info("p = {}", ptc.data().p1[j]);
  }
  ptc.sync_to_device();
  for (int i = 0; i < 3; i++) {
    rad.emit_photons(photons, ptc);
    ptc.sync_to_host();
    photons.sync_to_host();
    Logger::print_info("Current Particle momenta:");
    for (int j = 0; j < ptc.number(); j++) {
      Logger::print_info("p = {}", ptc.data().p1[j]);
    }
    Logger::print_info("-------");
  }
}
