#include "core/particles.h"
#include "core/photons.h"
#include "sim_environment.h"
#include "utils/logger.h"

using namespace Aperture;

int
main(int argc, char *argv[]) {
  sim_environment env(&argc, &argv);

  particles_t ptc(100, true);
  photons_t ph(100, true);
  int N1 = env.local_grid().mesh().dims[0];
  if (env.domain_info().rank == 0) {
    ptc.append({0.5, 0.5, 0.5}, {1.0, 0.0, 0.0}, 1 + 7 * N1,
               ParticleType::electron);
    ptc.append({0.5, 0.5, 0.5}, {2.0, 0.0, 0.0}, (N1 - 1) + 3 * N1,
               ParticleType::electron);
    ptc.append({0.5, 0.5, 0.5}, {3.0, 1.0, 0.0}, 1 + (N1 - 1) * N1,
               ParticleType::electron);
    ptc.append({0.5, 0.5, 0.5}, {4.0, -1.0, 0.0}, (N1 - 1) + 0 * N1,
               ParticleType::electron);
    ph.append({0.1, 0.2, 0.3}, {1.0, 1.0, 1.0}, 2 + 8 * N1, 0.0);
  }
  env.send_particles(ptc);
  env.send_particles(ph);
  ptc.sort_by_cell(env.local_grid());
  ph.sort_by_cell(env.local_grid());

  Logger::print_debug_all("Rank {} has {} particles:", env.domain_info().rank,
                          ptc.number());
  Logger::print_debug_all("Rank {} has {} photons:", env.domain_info().rank,
                          ph.number());
  for (unsigned int i = 0; i < ptc.number(); i++) {
    auto c = ptc.data().cell[i];
    Logger::print_debug_all("cell {}, {}", c % N1, c / N1);
  }

  return 0;
}
