#include "core/particles.h"
#include "sim_environment.h"
#include "utils/logger.h"

using namespace Aperture;

int
main(int argc, char *argv[]) {
  sim_environment env(&argc, &argv);

  particles_t ptc(100);
  int N1 = env.local_grid().mesh().dims[0];
  if (env.domain_info().rank == 0) {
    ptc.append({0.5, 0.5, 0.5}, {1.0, 0.0, 0.0}, 1 + 5 * N1,
               ParticleType::electron);
    ptc.append({0.5, 0.5, 0.5}, {2.0, 0.0, 0.0}, (N1 - 1) + 3 * N1,
               ParticleType::electron);
    ptc.append({0.5, 0.5, 0.5}, {3.0, 1.0, 0.0}, 1 + (N1 - 1) * N1,
               ParticleType::electron);
    ptc.append({0.5, 0.5, 0.5}, {4.0, -1.0, 0.0}, (N1 - 1) + 0 * N1,
               ParticleType::electron);
  }
  env.send_particles(ptc);
  ptc.sort_by_cell(env.local_grid());

  Logger::print_debug_all("Rank {} has {} particles", env.domain_info().rank,
                          ptc.number());

  return 0;
}
