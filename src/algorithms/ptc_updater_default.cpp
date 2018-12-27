#include "ptc_updater_default.h"
#include "sim_data.h"
#include "sim_environment.h"
#include "omp.h"

namespace Aperture {

ptc_updater_default::ptc_updater_default(const Environment &env)
    : ptc_updater(env) {}

ptc_updater_default::~ptc_updater_default() {}

void
ptc_updater_default::update_particles(sim_data &data, double dt) {}

void
ptc_updater_default::handle_boundary(sim_data &data) {}

}  // namespace Aperture
