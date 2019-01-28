#include "cuda/core/cu_sim_data1d.h"
#include "cuda/core/field_solver_1dgr.h"

namespace Aperture {

field_solver_1dgr_dev::field_solver_1dgr_dev(const Grid_1dGR_dev& g)
    : m_grid(g) {}

field_solver_1dgr_dev::~field_solver_1dgr_dev() {}

void
field_solver_1dgr_dev::update_fields(cu_sim_data1d& data, double dt,
                                     double time) {}

}  // namespace Aperture