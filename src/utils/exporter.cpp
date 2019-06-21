#include "exporter.h"
#include "sim_params.h"
#include "sim_data.h"
#include "sim_environment.h"

namespace Aperture {

exporter::exporter(sim_environment& env, uint32_t& timestep) :
    m_env(env) {
  auto& mesh = m_env.local_grid().mesh();
  m_grid_data.f.resize(mesh.extent_less());
}

exporter::~exporter() {}


}
