#include "algorithms/field_solver_1dgap.h"
#include "cuda/constant_mem.h"
#include "cuda/cudaUtility.h"
#include "cuda/utils/pitchptr.cuh"
#include "sim_data.h"
#include "sim_environment.h"
#include "utils/logger.h"
#include "cuda_runtime.h"

namespace Aperture {

namespace Kernels {

__global__ void
update_e_1dgap(pitchptr<Scalar> e1, pitchptr<Scalar> j1, Scalar j0,
               Scalar dt) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x +
                  dev_mesh.guard[0] + 1;
       i < dev_mesh.dims[0] - dev_mesh.guard[0] - 1;
       i += blockDim.x * gridDim.x) {
    e1(i, 0) += (j0 - j1(i, 0)) * dt;
  }
}

}  // namespace Kernels

field_solver_1dgap::field_solver_1dgap(sim_environment& env)
    : m_env(env) {}

field_solver_1dgap::~field_solver_1dgap() {}

void
field_solver_1dgap::update_fields(sim_data& data, double dt,
                                  double time) {
  Logger::print_info("Updating fields");

  Scalar j0 = m_env.params().B0 * 1.6;
  Kernels::update_e_1dgap<<<256, 512>>>(get_pitchptr(data.E.data(0)),
                                        get_pitchptr(data.J.data(0)),
                                        j0, dt);
  CudaCheckError();
}

}  // namespace Aperture
