#include "algorithms/field_solver_1dgr.h"
#include "cuda/constant_mem.h"
#include "cuda/cudaUtility.h"
#include "cuda/utils/pitchptr.h"
#include "sim_data.h"
#include "sim_environment.h"
#include "utils/logger.h"

namespace Aperture {

namespace Kernels {

__global__ void
update_e_1dgr(pitchptr<Scalar> e1, pitchptr<Scalar> e3,
              pitchptr<Scalar> j1, Grid_1dGR::mesh_ptrs mesh_ptrs,
              Scalar dt) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x +
                  dev_mesh.guard[0] + 1;
       i < dev_mesh.dims[0] - dev_mesh.guard[0] - 5;
       i += blockDim.x * gridDim.x) {
    // Scalar j0 = mesh_ptrs.j0[i] * mesh_ptrs.dpsidth[i];
    Scalar j0 = mesh_ptrs.j0[i];
    // Scalar r = dev_mesh.pos(0, i, 0);
    // *ptrAddr(e1, i, 0) += (j0 - *ptrAddr(j1, i, 0)) * dt;
    // Scalar D1 = e1(i, 0);
    // D1 *= mesh_ptrs.K1_j[i];
    // D1 += mesh_ptrs.K1_j[i] * (j0 - j1(i, 0)) * dt;
    e1(i, 0) += (j0 - j1(i, 0)) * dt;
    // printf("E1 is %f\n", *ptrAddr(e1, i, 0));
    // TODO: Check all equations
    // *ptrAddr(e3, i, 0) +=
    //     ((j0 - *ptrAddr(j1, i, 0)) * *ptrAddr(b3, i, 0) /
    //          *ptrAddr(b1, i, 0) +
    //      dev_params.omega * (mesh_ptrs.rho0[i] - *ptrAddr(rho, i,
    //      0))) *
    //     dt;
  }
}

}  // namespace Kernels

field_solver_1dgr::field_solver_1dgr(sim_environment& env)
    : m_env(env) {}

field_solver_1dgr::~field_solver_1dgr() {}

void
field_solver_1dgr::update_fields(sim_data& data, double dt,
                                 double time) {
  Logger::print_info("Updating fields");

  const Grid_1dGR& grid =
      *dynamic_cast<const Grid_1dGR*>(&m_env.local_grid());
  auto mesh_ptrs = grid.get_mesh_ptrs();
  Kernels::update_e_1dgr<<<256, 512>>>(
      // data.E.ptr(0), data.E.ptr(2), data.J.ptr(0),
      get_pitchptr(data.E.data(0)), get_pitchptr(data.E.data(2)),
      get_pitchptr(data.J.data(0)), mesh_ptrs, dt);
  CudaCheckError();
}

}  // namespace Aperture
