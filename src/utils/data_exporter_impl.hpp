#ifndef __DATA_EXPORTER_IMPL_H_
#define __DATA_EXPORTER_IMPL_H_

#include "data_exporter.h"
#include "sim_params.h"
#include "sim_data.h"
#include "sim_environment.h"

namespace Aperture {

data_exporter::data_exporter(sim_environment& env, uint32_t& timestep) :
    m_env(env) {
  auto& mesh = m_env.local_grid().mesh();
  auto ext = mesh.extent_less();
  auto d = m_env.params().downsample;
  tmp_grid_data.resize(ext.width() / d, ext.height() / d,
                       ext.depth() / d);
}

data_exporter::~data_exporter() {}


}


#endif // __DATA_EXPORTER_IMPL_H_
