#include "algorithms/field_solver_default.h"
#include "algorithms/field_solver_helper.cuh"
#include "algorithms/finite_diff.h"
#include "cuda/constant_mem.h"
#include "cuda/cudaUtility.h"
#include "data/detail/multi_array_utils.hpp"
#include "data/fields_utils.h"
#include "utils/timer.h"
// #include "sim_environment_dev.h"

namespace Aperture {

namespace Kernels {

// TODO: work out the cuda kernels for the 3D finite difference

__global__ void
update_field_1d_simple(Scalar *E, const Scalar *J, const Scalar *J_b) {
  auto dt = dev_params.delta_t;
  for (int i = dev_params.guard[0] - 1 + blockIdx.x * blockDim.x +
               threadIdx.x;
       i < dev_params.guard[0] + dev_params.N[0];
       i += blockDim.x * gridDim.x) {
    if (i >= dev_params.guard[0] - 1 &&
        i < dev_params.guard[0] + dev_params.N[0])
      E[i] += dt * (J_b[i] - J[i]);
  }
}

}  // namespace Kernels

FieldSolver_Default::FieldSolver_Default(const Grid &g)
    : m_dE(g), m_dB(g), m_background_j(g) {
  m_background_j.initialize();
}

FieldSolver_Default::~FieldSolver_Default() {}

void
FieldSolver_Default::update_fields(vfield_t &E, vfield_t &B,
                                   const vfield_t &J, double dt,
                                   double time) {
  Logger::print_info("Updating fields");
  auto &grid = E.grid();
  auto &mesh = grid.mesh();
  // Explicit update
  if (grid.dim() == 1) {
    Kernels::update_field_1d_simple<<<512, 512>>>(
        E.data(0).data(), J.data(0).data(),
        m_background_j.data(0).data());
    CudaCheckError();
  } else if (grid.dim() == 3) {
    // Compute the curl of E and add it to B
    curl_add(B, E, dt);
    cudaDeviceSynchronize();

    // Compute the update of E
    curl_add(E, B, dt);
    field_add(E, J, -dt);
    cudaDeviceSynchronize();
  }

  if (m_comm_callback_vfield != nullptr) {
    m_comm_callback_vfield(E);
    if (grid.dim() > 1) {
      m_comm_callback_vfield(B);
    }
  }
}

void
FieldSolver_Default::update_fields(Aperture::SimData &data, double dt,
                                   double time) {
  update_fields(data.E, data.B, data.J, dt, time);
}

void
FieldSolver_Default::set_background_j(const vfield_t &j) {
  m_background_j = j;
  m_background_j.sync_to_device();
  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
}

}  // namespace Aperture
