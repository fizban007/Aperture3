#include "algorithms/field_solver_helper.cuh"
#include "algorithms/field_solver_log_sph.h"
#include "algorithms/finite_diff.h"
#include "cuda/constant_mem.h"
#include "cuda/cudaUtility.h"
#include "data/detail/multi_array_utils.hpp"
#include "data/fields_utils.h"
#include "utils/timer.h"

namespace Aperture {

namespace Kernels {

template <int DIM1, int DIM2>
__global__ void
compute_e_update(cudaPitchedPtr e1, cudaPitchedPtr e2,
                 cudaPitchedPtr e3, cudaPitchedPtr b1,
                 cudaPitchedPtr b2, cudaPitchedPtr b3,
                 cudaPitchedPtr j1, cudaPitchedPtr j2,
                 cudaPitchedPtr j3, Grid_LogSph::mesh_ptrs mesh_ptrs,
                 Scalar dt) {
  // Declare cache array in shared memory
  // __shared__ Scalar
  //     s_b1[DIM2 + 2 * Pad<2>::val][DIM1 + 2 * Pad<2>::val];
  // __shared__ Scalar
  //     s_b2[DIM2 + 2 * Pad<2>::val][DIM1 + 2 * Pad<2>::val];
  // __shared__ Scalar
  //     s_b3[DIM2 + 2 * Pad<2>::val][DIM1 + 2 * Pad<2>::val];

  // Load shared memory
  int t1 = blockIdx.x, t2 = blockIdx.y;
  int c1 = threadIdx.x + Pad<2>::val, c2 = threadIdx.y + Pad<2>::val;
  int n1 = dev_mesh.guard[0] + t1 * DIM1 + c1 - Pad<2>::val;
  int n2 = dev_mesh.guard[1] + t2 * DIM2 + c2 - Pad<2>::val;
  size_t globalOffset = n2 * e1.pitch + n1 * sizeof(Scalar);

  // init_shared_memory<2, DIM1, DIM2>(s_u1, s_u2, s_u3, u1, u2, u3,
  //                                   globalOffset, c1, c2);
  // __syncthreads();

  // Do the actual computation here
  // (Curl u)_1 = d2u3 - d3u2
  (*ptrAddr(e1, globalOffset)) +=
      dt * ((*ptrAddr(b3, globalOffset) *
                 *ptrAddr(mesh_ptrs.l3_b, globalOffset) -
             *ptrAddr(b3, globalOffset - e1.pitch) *
                 *ptrAddr(mesh_ptrs.l3_b, globalOffset - e1.pitch)) /
                *ptrAddr(mesh_ptrs.A1_e, globalOffset) -
            *ptrAddr(j1, globalOffset));
  // q * d2<Order, DIM1, DIM2>(s_u3, c1, c2 + flip(s3[1]));

  // (Curl u)_2 = d3u1 - d1u3
  // (*(Scalar*)((char*)e2.ptr + globalOffset)) += 0.0;
  // q * d1<Order, DIM1, DIM2>(s_u3, c1 + flip(s3[0]), c2);
  (*ptrAddr(e2, globalOffset)) +=
      dt *
      ((*ptrAddr(b3, globalOffset) *
            *ptrAddr(mesh_ptrs.l3_b, globalOffset) -
        *ptrAddr(b3, globalOffset - sizeof(Scalar)) *
            *ptrAddr(mesh_ptrs.l3_b, globalOffset - sizeof(Scalar))) /
           *ptrAddr(mesh_ptrs.A2_e, globalOffset) -
       *ptrAddr(j2, globalOffset));

  // (Curl u)_3 = d1u2 - d2u1
  // (*(Scalar*)((char*)e3.ptr + globalOffset)) += 0.0;
  // q * (d1<Order, DIM1, DIM2>(s_u2, c1 + flip(s2[0]), c2) -
  //      d2<Order, DIM1, DIM2>(s_u1, c1, c2 + flip(s1[1])));
  (*ptrAddr(e3, globalOffset)) +=
      dt *
      ((*ptrAddr(b2, globalOffset) *
            *ptrAddr(mesh_ptrs.l2_b, globalOffset) -
        *ptrAddr(b2, globalOffset - sizeof(Scalar)) *
            *ptrAddr(mesh_ptrs.l2_b, globalOffset - sizeof(Scalar)) -
        *ptrAddr(b1, globalOffset - e1.pitch) *
            *ptrAddr(mesh_ptrs.l1_b, globalOffset - e1.pitch) +
        *ptrAddr(b1, globalOffset) *
            *ptrAddr(mesh_ptrs.l1_b, globalOffset)) /
           *ptrAddr(mesh_ptrs.A3_e, globalOffset) -
       *ptrAddr(j3, globalOffset));
}

template <int DIM1, int DIM2>
__global__ void
compute_b_update(cudaPitchedPtr e1, cudaPitchedPtr e2,
                 cudaPitchedPtr e3, cudaPitchedPtr b1,
                 cudaPitchedPtr b2, cudaPitchedPtr b3,
                 Grid_LogSph::mesh_ptrs mesh_ptrs, Scalar dt) {
  // Declare cache array in shared memory
  // __shared__ Scalar
  //     s_b1[DIM2 + 2 * Pad<2>::val][DIM1 + 2 * Pad<2>::val];
  // __shared__ Scalar
  //     s_b2[DIM2 + 2 * Pad<2>::val][DIM1 + 2 * Pad<2>::val];
  // __shared__ Scalar
  //     s_b3[DIM2 + 2 * Pad<2>::val][DIM1 + 2 * Pad<2>::val];

  // Load shared memory
  int t1 = blockIdx.x, t2 = blockIdx.y;
  int c1 = threadIdx.x + Pad<2>::val, c2 = threadIdx.y + Pad<2>::val;
  int n1 = dev_mesh.guard[0] + t1 * DIM1 + c1 - Pad<2>::val;
  int n2 = dev_mesh.guard[1] + t2 * DIM2 + c2 - Pad<2>::val;
  size_t globalOffset = n2 * e1.pitch + n1 * sizeof(Scalar);

  // init_shared_memory<2, DIM1, DIM2>(s_u1, s_u2, s_u3, u1, u2, u3,
  //                                   globalOffset, c1, c2);
  // __syncthreads();

  // Do the actual computation here
  // (Curl u)_1 = d2u3 - d3u2
  (*ptrAddr(b1, globalOffset)) +=
      -dt *
      (*ptrAddr(e3, globalOffset + e1.pitch) *
           *ptrAddr(mesh_ptrs.l3_e, globalOffset + e1.pitch) -
       *ptrAddr(e3, globalOffset) *
           *ptrAddr(mesh_ptrs.l3_e, globalOffset)) /
      *ptrAddr(mesh_ptrs.A1_b, globalOffset);
  // q * d2<Order, DIM1, DIM2>(s_u3, c1, c2 + flip(s3[1]));

  // (Curl u)_2 = d3u1 - d1u3
  // (*(Scalar*)((char*)e2.ptr + globalOffset)) += 0.0;
  // q * d1<Order, DIM1, DIM2>(s_u3, c1 + flip(s3[0]), c2);
  (*ptrAddr(b2, globalOffset)) +=
      -dt *
      (*ptrAddr(e3, globalOffset + sizeof(Scalar)) *
           *ptrAddr(mesh_ptrs.l3_e, globalOffset + sizeof(Scalar)) -
       *ptrAddr(e3, globalOffset) *
           *ptrAddr(mesh_ptrs.l3_e, globalOffset)) /
      *ptrAddr(mesh_ptrs.A2_b, globalOffset);

  // (Curl u)_3 = d1u2 - d2u1
  // (*(Scalar*)((char*)e3.ptr + globalOffset)) += 0.0;
  // q * (d1<Order, DIM1, DIM2>(s_u2, c1 + flip(s2[0]), c2) -
  //      d2<Order, DIM1, DIM2>(s_u1, c1, c2 + flip(s1[1])));
  (*ptrAddr(b3, globalOffset)) +=
      -dt *
      ((*ptrAddr(e2, globalOffset + sizeof(Scalar)) *
            *ptrAddr(mesh_ptrs.l2_e, globalOffset + sizeof(Scalar)) -
        *ptrAddr(e2, globalOffset) *
            *ptrAddr(mesh_ptrs.l2_e, globalOffset) -
        *ptrAddr(e1, globalOffset) *
            *ptrAddr(mesh_ptrs.l1_e, globalOffset) +
        *ptrAddr(e1, globalOffset + e1.pitch) *
            *ptrAddr(mesh_ptrs.l1_e, globalOffset + e1.pitch)) /
       *ptrAddr(mesh_ptrs.A3_b, globalOffset));
}

}  // namespace Kernels

FieldSolver_LogSph::FieldSolver_LogSph(const Grid_LogSph& g)
    : m_grid(g) {}

FieldSolver_LogSph::~FieldSolver_LogSph() {}

void
FieldSolver_LogSph::update_fields(SimData& data, double dt,
                                  double time) {}

void
FieldSolver_LogSph::update_fields(vfield_t& E, vfield_t& B,
                                  const vfield_t& J, double dt,
                                  double time) {
  Logger::print_info("Updating fields");
  auto mesh_ptrs = m_grid.get_mesh_ptrs();
  auto& mesh = m_grid.mesh();

  if (m_grid.dim() == 2) {
    // We only implemented 2d at the moment
    dim3 blockSize(32, 16);
    dim3 gridSize(mesh.reduced_dim(0) / 32, mesh.reduced_dim(1) / 16);
    // Update B
    Kernels::compute_b_update<32, 16><<<gridSize, blockSize>>>(
        E.ptr(0), E.ptr(1), E.ptr(2), B.ptr(0), B.ptr(1), B.ptr(2),
        mesh_ptrs, dt);
    CudaCheckError();

    // Update E
    Kernels::compute_e_update<32, 16><<<gridSize, blockSize>>>(
        E.ptr(0), E.ptr(1), E.ptr(2), B.ptr(0), B.ptr(1), B.ptr(2),
        J.ptr(0), J.ptr(1), J.ptr(2), mesh_ptrs, dt);
    CudaCheckError();

    if (m_comm_callback_vfield != nullptr) {
      m_comm_callback_vfield(E);
      m_comm_callback_vfield(B);
    }
  }

}

void
FieldSolver_LogSph::set_background_j(const vfield_t& J) {}

}  // namespace Aperture