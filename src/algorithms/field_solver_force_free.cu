#include "algorithms/field_solver_force_free.h"
#include "algorithms/finite_diff.h"
#include "algorithms/field_solver_helper.cuh"
#include "data/fields_utils.h"
#include "cuda/cudaUtility.h"
#include "cuda/constant_mem.h"

namespace Aperture {

namespace Kernels {

template <int DIM1, int DIM2, int DIM3>
__global__
void compute_FFE_EdotB(cudaPitchedPtr eb,
                       cudaPitchedPtr e1, cudaPitchedPtr e2, cudaPitchedPtr e3,
                       cudaPitchedPtr b1, cudaPitchedPtr b2, cudaPitchedPtr b3, Scalar q) {
  // Declare cache array in shared memory
  __shared__ Scalar s_e1[DIM3 + 2*Pad<2>::val]
      [DIM2 + 2*Pad<2>::val][DIM1 + 2*Pad<2>::val];
  __shared__ Scalar s_e2[DIM3 + 2*Pad<2>::val]
      [DIM2 + 2*Pad<2>::val][DIM1 + 2*Pad<2>::val];
  __shared__ Scalar s_e3[DIM3 + 2*Pad<2>::val]
      [DIM2 + 2*Pad<2>::val][DIM1 + 2*Pad<2>::val];
  __shared__ Scalar s_b1[DIM3 + 2*Pad<2>::val]
      [DIM2 + 2*Pad<2>::val][DIM1 + 2*Pad<2>::val];
  __shared__ Scalar s_b2[DIM3 + 2*Pad<2>::val]
      [DIM2 + 2*Pad<2>::val][DIM1 + 2*Pad<2>::val];
  __shared__ Scalar s_b3[DIM3 + 2*Pad<2>::val]
      [DIM2 + 2*Pad<2>::val][DIM1 + 2*Pad<2>::val];

  // Load shared memory
  int t1 = blockIdx.x, t2 = blockIdx.y, t3 = blockIdx.z;
  int c1 = threadIdx.x + Pad<2>::val,
      c2 = threadIdx.y + Pad<2>::val,
      c3 = threadIdx.z + Pad<2>::val;
  size_t globalOffset =  (dev_mesh.guard[2] + t3 * DIM3 + c3 - Pad<2>::val) * e1.pitch * e1.ysize +
                         (dev_mesh.guard[1] + t2 * DIM2 + c2 - Pad<2>::val) * e1.pitch +
                         (dev_mesh.guard[0] + t1 * DIM1 + c1 - Pad<2>::val) * sizeof(Scalar);

  init_shared_memory<2, DIM1, DIM2, DIM3>(s_e1, s_e2, s_e3, e1, e2, e3,
                                          globalOffset, c1, c2, c3);
  init_shared_memory<2, DIM1, DIM2, DIM3>(s_b1, s_b2, s_b3, b1, b2, b3,
                                          globalOffset, c1, c2, c3);
  __syncthreads();

  Scalar vecE1 = 0.5f * (s_e1[c3][c2][c1] + s_e1[c3][c2][c1 - 1]);
  Scalar vecE2 = 0.5f * (s_e2[c3][c2][c1] + s_e2[c3][c2 - 1][c1]);
  Scalar vecE3 = 0.5f * (s_e3[c3][c2][c1] + s_e3[c3 - 1][c2][c1]);
  Scalar vecB1 = 0.25f * (s_b1[c3][c2][c1] + s_b1[c3 - 1][c2][c1] +
                          s_b1[c3][c2 - 1][c1] + s_b1[c3 - 1][c2 - 1][c1]);
  Scalar vecB2 = 0.25f * (s_b2[c3][c2][c1] + s_b2[c3 - 1][c2][c1] +
                          s_b2[c3][c2][c1 - 1] + s_b2[c3 - 1][c2][c1 - 1]);
  Scalar vecB3 = 0.25f * (s_b3[c3][c2][c1] + s_b3[c3][c2][c1 - 1] +
                          s_b3[c3][c2 - 1][c1] + s_b3[c3][c2 - 1][c1 - 1]);
  Scalar EdotB = vecE1 * vecB1 + vecE2 * vecB2 + vecE3 * vecB3;

  // Do the actual computation here
  (*(Scalar*)((char*)eb.ptr + globalOffset)) += q * EdotB;
}

template <int DIM1, int DIM2, int DIM3>
__global__
void compute_FFE_J(cudaPitchedPtr j1, cudaPitchedPtr j2, cudaPitchedPtr j3,
                   cudaPitchedPtr e1, cudaPitchedPtr e2, cudaPitchedPtr e3,
                   cudaPitchedPtr b1, cudaPitchedPtr b2, cudaPitchedPtr b3,
                   cudaPitchedPtr f, Scalar q) {
  // Declare cache array in shared memory
  __shared__ Scalar s_e1[DIM3 + 2*Pad<2>::val]
      [DIM2 + 2*Pad<2>::val][DIM1 + 2*Pad<2>::val];
  __shared__ Scalar s_e2[DIM3 + 2*Pad<2>::val]
      [DIM2 + 2*Pad<2>::val][DIM1 + 2*Pad<2>::val];
  __shared__ Scalar s_e3[DIM3 + 2*Pad<2>::val]
      [DIM2 + 2*Pad<2>::val][DIM1 + 2*Pad<2>::val];
  __shared__ Scalar s_b1[DIM3 + 2*Pad<2>::val]
      [DIM2 + 2*Pad<2>::val][DIM1 + 2*Pad<2>::val];
  __shared__ Scalar s_b2[DIM3 + 2*Pad<2>::val]
      [DIM2 + 2*Pad<2>::val][DIM1 + 2*Pad<2>::val];
  __shared__ Scalar s_b3[DIM3 + 2*Pad<2>::val]
      [DIM2 + 2*Pad<2>::val][DIM1 + 2*Pad<2>::val];
  __shared__ Scalar s_f[DIM3 + 2*Pad<2>::val]
      [DIM2 + 2*Pad<2>::val][DIM1 + 2*Pad<2>::val];

  // Load shared memory
  int t1 = blockIdx.x, t2 = blockIdx.y, t3 = blockIdx.z;
  int c1 = threadIdx.x + Pad<2>::val,
      c2 = threadIdx.y + Pad<2>::val,
      c3 = threadIdx.z + Pad<2>::val;
  size_t globalOffset =  (dev_mesh.guard[2] + t3 * DIM3 + c3 - Pad<2>::val) * e1.pitch * e1.ysize +
                         (dev_mesh.guard[1] + t2 * DIM2 + c2 - Pad<2>::val) * e1.pitch +
                         (dev_mesh.guard[0] + t1 * DIM1 + c1 - Pad<2>::val) * sizeof(Scalar);

  init_shared_memory<2, DIM1, DIM2, DIM3>(s_e1, s_e2, s_e3, e1, e2, e3,
                                          globalOffset, c1, c2, c3);
  init_shared_memory<2, DIM1, DIM2, DIM3>(s_b1, s_b2, s_b3, b1, b2, b3,
                                          globalOffset, c1, c2, c3);
  init_shared_memory<2, DIM1, DIM2, DIM3>(s_f, f, globalOffset, c1, c2, c3);
  __syncthreads();

  Scalar vecE1 = 0.5f * (s_e1[c3][c2][c1] + s_e1[c3][c2][c1 - 1]);
  Scalar vecE2 = 0.5f * (s_e2[c3][c2][c1] + s_e2[c3][c2 - 1][c1]);
  Scalar vecE3 = 0.5f * (s_e3[c3][c2][c1] + s_e3[c3 - 1][c2][c1]);
  Scalar vecB1 = 0.25f * (s_b1[c3][c2][c1] + s_b1[c3 - 1][c2][c1] +
                          s_b1[c3][c2 - 1][c1] + s_b1[c3 - 1][c2 - 1][c1]);
  Scalar vecB2 = 0.25f * (s_b2[c3][c2][c1] + s_b2[c3 - 1][c2][c1] +
                          s_b2[c3][c2][c1 - 1] + s_b2[c3 - 1][c2][c1 - 1]);
  Scalar vecB3 = 0.25f * (s_b3[c3][c2][c1] + s_b3[c3][c2][c1 - 1] +
                          s_b3[c3][c2 - 1][c1] + s_b3[c3][c2 - 1][c1 - 1]);
  Scalar inv_B_sqr = 1.0f / (vecB1 * vecB1 + vecB2 * vecB2 + vecB3 * vecB3);
  Scalar divE = (s_e1[c3][c2][c1] - s_e1[c3][c2][c1 - 1]) / dev_mesh.delta[0] +
                (s_e2[c3][c2][c1] - s_e2[c3][c2 - 1][c1]) / dev_mesh.delta[1] +
                (s_e3[c3][c2][c1] - s_e3[c3 - 1][c2][c1]) / dev_mesh.delta[2];
  Scalar EcrossB1 = vecE2 * vecB3 - vecE3 * vecB2;
  Scalar EcrossB2 = vecE3 * vecB1 - vecE1 * vecB3;
  Scalar EcrossB3 = vecE1 * vecB2 - vecE2 * vecB1;
  // Scalar EdotB = vecE1 * vecB1 + vecE2 * vecB2 + vecE3 * vecB3;

  // Do the actual computation here
  (*(Scalar*)((char*)j1.ptr + globalOffset)) = q * (s_f[c3][c2][c1] * vecB1 + divE * EcrossB1) * inv_B_sqr;
  (*(Scalar*)((char*)j2.ptr + globalOffset)) = q * (s_f[c3][c2][c1] * vecB2 + divE * EcrossB2) * inv_B_sqr;
  (*(Scalar*)((char*)j3.ptr + globalOffset)) = q * (s_f[c3][c2][c1] * vecB3 + divE * EcrossB3) * inv_B_sqr;
}



}

FieldSolver_FFE::FieldSolver_FFE(const Grid& g) :
    m_sf(g), m_tmp(g), m_tmp2(g),
    m_e1(g), m_e2(g), m_e3(g), m_e4(g),
    m_b1(g), m_b2(g), m_b3(g), m_b4(g) {
  // m_j1(g), m_j2(g), m_j3(g), m_j4(g) {
  m_b1.set_field_type(FieldType::B);
  m_b2.set_field_type(FieldType::B);
  m_b3.set_field_type(FieldType::B);
  m_b4.set_field_type(FieldType::B);
}

FieldSolver_FFE::~FieldSolver_FFE() {}

void
FieldSolver_FFE::update_fields(SimData &data, double dt, double time) {
  
}

void
FieldSolver_FFE::compute_J(vfield_t &J, const vfield_t &E, const vfield_t &B) {
  
}

void
FieldSolver_FFE::update_field_substep(vfield_t &E_out, vfield_t &B_out, vfield_t &J_out,
                                      const vfield_t &E_in, const vfield_t &B_in, Scalar dt) {
  // Initialize all tmp fields to zero on the device
  m_tmp.initialize();
  // m_tmp2.initialize();
  m_sf.initialize();

  // Compute the curl of E_in and set it to m_tmp
  m_tmp.set_field_type(FieldType::B);
  curl(m_tmp, E_in);
  // m_tmp2 is now equal to curl E_in
  field_add(B_out, m_tmp, dt);
  ffe_edotb(m_sf, E_in, m_tmp, 1.0f);

  m_tmp.initialize();
  // Compute the curl of B_in and set it to m_tmp
  m_tmp.set_field_type(FieldType::E);
  curl(m_tmp, B_in);
  // m_tmp is now equal to curl B_in
  field_add(E_out, m_tmp, dt);
  ffe_edotb(m_sf, m_tmp, B_in, -1.0f);

  // Now compute FFE current at the cell center
  ffe_j(m_tmp, m_sf, E_in, B_in);
  // interpolate J back to staggered position
  m_tmp.interpolate_from_center(J_out);
  // Add J to E_out, conclude the substep update
  field_add(E_out, J_out, dt);

  // TODO: Figure out how to best handle removal of the parallel delta_E
}

void
FieldSolver_FFE::ffe_edotb(ScalarField<Scalar>& result, const VectorField<Scalar>& E,
                           const VectorField<Scalar>& B, Scalar q) {
  auto& grid = E.grid();
  auto& mesh = grid.mesh();

  dim3 blockSize(16, 8, 8);
  dim3 gridSize(mesh.reduced_dim(0) / 16, mesh.reduced_dim(1) / 8,
                mesh.reduced_dim(2) / 8);
  Kernels::compute_FFE_EdotB<16, 8, 8><<<gridSize, blockSize>>>
      (result.ptr(), E.ptr(0), E.ptr(1), E.ptr(2),
       B.ptr(0), B.ptr(1), B.ptr(2), q);
  CudaCheckError();
}

void
FieldSolver_FFE::ffe_j(VectorField<Scalar>& result, const ScalarField<Scalar>& tmp_f,
                       const VectorField<Scalar>& E, const VectorField<Scalar>& B,
                       Scalar q) {
  auto& grid = E.grid();
  auto& mesh = grid.mesh();

  dim3 blockSize(16, 8, 4);
  dim3 gridSize(mesh.reduced_dim(0) / 16, mesh.reduced_dim(1) / 8,
                mesh.reduced_dim(2) / 4);

  Kernels::compute_FFE_J<16, 8, 4><<<gridSize, blockSize>>>
      (result.ptr(0), result.ptr(1), result.ptr(2),
       E.ptr(0), E.ptr(1), E.ptr(2),
       B.ptr(0), B.ptr(1), B.ptr(2),
       tmp_f.ptr(), q);
  CudaCheckError();
}

}
