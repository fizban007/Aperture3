#include "ptc_updater_default.h"
#include "sim_data.h"
#include "sim_environment.h"
#include "omp.h"

#include <immintrin.h>

namespace Aperture {

ptc_updater_default::ptc_updater_default(const Environment &env)
    : ptc_updater(env) {}

ptc_updater_default::~ptc_updater_default() {}

void
ptc_updater_default::update_particles(sim_data &data, double dt) {
  auto& ptc = data.particles;
  if (ptc.number() > 0) {
    // TODO: Vay push
    for (size_t idx = 0; idx < ptc.number(); idx++) {
      
    }

    // TODO: Current deposit

    // TODO: Push photons as well
  }

}

#if defined(__AVX2__)
#endif

void
ptc_updater_default::handle_boundary(sim_data &data) {}

}  // namespace Aperture
