#include "radiation/radiation_field.h"
#include "sim_environment_dev.h"

namespace Aperture {

RadiationField::RadiationField(const Environment& env)
    : m_env(env),
      m_data(env.params().rad_energy_bins,
             env.local_grid().extent()[0]) {}

RadiationField::~RadiationField() {}

void
RadiationField::advect(Scalar dt) {}

}  // namespace Aperture
