#include "radiation/radiation_field.h"
#include "cu_sim_environment.h"

namespace Aperture {

RadiationField::RadiationField(const cu_sim_environment& env)
    : m_env(env),
      m_data(env.params().rad_energy_bins,
             env.local_grid().extent()[0]) {}

RadiationField::~RadiationField() {}

void
RadiationField::advect(Scalar dt) {}

}  // namespace Aperture
