#include "algorithms/field_solver_default.h"
#include "data/detail/multi_array_utils.hpp"
#include "cuda/cudaUtility.h"
#include "cuda/constant_mem.h"
// #include "sim_environment.h"

namespace Aperture {

namespace Kernels {

__global__ void update_field(Scalar* E, const Scalar* J, const Scalar* J_b) {
  auto dt = dev_params.delta_t;
  for (int i = dev_params.guard[0] - 1 + blockIdx.x * blockDim.x + threadIdx.x;
       i < dev_params.guard[0] + dev_params.N[0];
       i += blockDim.x * gridDim.x) {
    E[i] += dt * (J_b[i] - J[i]);
  }
}

}

FieldSolver_Default::FieldSolver_Default(const Grid& g)
    : m_dE(g), m_dB(g), m_background_j(g) {
  m_background_j.initialize();
}

FieldSolver_Default::~FieldSolver_Default() {}

void
FieldSolver_Default::update_fields(vfield_t &E, vfield_t &B, const vfield_t &J,
                                   double dt, double time) {
  // Logger::print_info("Updating fields");
  auto &grid = E.grid();
  auto &mesh = grid.mesh();
  // Explicit update
  if (grid.dim() == 1) {
    Kernels::update_field<<<512, 512>>>(E.ptr(0), J.ptr(0), m_background_j.ptr(0));
    CudaCheckError();
  }

  if (m_comm_callback_vfield != nullptr) {
    m_comm_callback_vfield(E);
  }
}

void
FieldSolver_Default::update_fields(Aperture::SimData &data, double dt,
                                    double time) {
  update_fields(data.E, data.B, data.J, dt, time);
  Kernels::map_array_binary_op<<<256, 256>>>
      (data.E.ptr(0), data.E.ptr(1), data.E.grid().extent(), detail::Op_PlusAssign<Scalar>());
  CudaCheckError();
}

void
FieldSolver_Default::set_background_j(const vfield_t &j) {
  m_background_j = j;
}



}