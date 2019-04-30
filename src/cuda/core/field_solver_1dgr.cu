#include "cuda/constant_mem.h"
#include "cuda/core/cu_sim_data1d.h"
#include "cuda/core/field_solver_1dgr.h"
#include "cuda/cudaUtility.h"
// #include "cuda/ptr_util.h"
#include "cuda/utils/pitchptr.cuh"

namespace Aperture {

namespace Kernels {

__global__ void
update_e_1dgr(pitchptr<Scalar> e1, pitchptr<Scalar> e3,
              pitchptr<Scalar> j1, pitchptr<Scalar> rho,
              Grid_1dGR_dev::mesh_ptrs mesh_ptrs, Scalar dt) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
       i < dev_mesh.dims[0]; i += blockDim.x * gridDim.x) {
    // Scalar j0 = mesh_ptrs.j0[i] * mesh_ptrs.dpsidth[i];
    Scalar j0 = mesh_ptrs.j0[i];
    // *ptrAddr(e1, i, 0) += (j0 - *ptrAddr(j1, i, 0)) * dt;
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

field_solver_1dgr_dev::field_solver_1dgr_dev(const Grid_1dGR_dev& g)
    : m_grid(g) {}

field_solver_1dgr_dev::~field_solver_1dgr_dev() {}

void
field_solver_1dgr_dev::update_fields(cu_sim_data1d& data, double dt,
                                     double time) {
  Logger::print_info("Updating fields");
  auto mesh_ptrs = m_grid.get_mesh_ptrs();
  Kernels::update_e_1dgr<<<256, 512>>>(data.E.ptr(0), data.E.ptr(2),
                                       data.J.ptr(0), data.Rho[0].ptr(),
                                       mesh_ptrs, dt);
  CudaCheckError();
}

}  // namespace Aperture
