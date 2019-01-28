#include "cuda/core/cu_sim_data1d.h"
#include "cuda/core/field_solver_1dgr.h"

namespace Aperture {

namespace Kernels {

__global__ void
update_e_1dgr(cudaPitchedPtr e1, cudaPitchedPtr e3,
              Grid_1dGR_dev::mesh_ptrs mesh_ptrs, Scalar dt) {}

}  // namespace Kernels

field_solver_1dgr_dev::field_solver_1dgr_dev(const Grid_1dGR_dev& g)
    : m_grid(g) {}

field_solver_1dgr_dev::~field_solver_1dgr_dev() {}

void
field_solver_1dgr_dev::update_fields(cu_sim_data1d& data, double dt,
                                     double time) {}

}  // namespace Aperture