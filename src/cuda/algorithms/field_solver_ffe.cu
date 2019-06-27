#include "algorithms/field_solver_ffe.h"
#include "cuda/constant_mem.h"
#include "cuda/core/field_solver_helper.cuh"
#include "cuda/core/finite_diff.h"
#include "cuda/cudaUtility.h"
#include "cuda/data/fields_utils.h"
#include "cuda/data_ptrs.h"
#include "cuda/utils/pitchptr.cuh"
#include "sim_data.h"
#include "sim_environment.h"
#include "utils/timer.h"

namespace Aperture {

namespace Kernels {

template <int DIM1, int DIM2, int DIM3>
__global__ void
compute_FFE_EdotB(cudaPitchedPtr eb, cudaPitchedPtr e1,
                  cudaPitchedPtr e2, cudaPitchedPtr e3,
                  cudaPitchedPtr b1, cudaPitchedPtr b2,
                  cudaPitchedPtr b3, Scalar q) {
  // Declare cache array in shared memory
  __shared__ Scalar s_e1[DIM3 + 2 * Pad<2>::val][DIM2 + 2 * Pad<2>::val]
                        [DIM1 + 2 * Pad<2>::val];
  __shared__ Scalar s_e2[DIM3 + 2 * Pad<2>::val][DIM2 + 2 * Pad<2>::val]
                        [DIM1 + 2 * Pad<2>::val];
  __shared__ Scalar s_e3[DIM3 + 2 * Pad<2>::val][DIM2 + 2 * Pad<2>::val]
                        [DIM1 + 2 * Pad<2>::val];
  __shared__ Scalar s_b1[DIM3 + 2 * Pad<2>::val][DIM2 + 2 * Pad<2>::val]
                        [DIM1 + 2 * Pad<2>::val];
  __shared__ Scalar s_b2[DIM3 + 2 * Pad<2>::val][DIM2 + 2 * Pad<2>::val]
                        [DIM1 + 2 * Pad<2>::val];
  __shared__ Scalar s_b3[DIM3 + 2 * Pad<2>::val][DIM2 + 2 * Pad<2>::val]
                        [DIM1 + 2 * Pad<2>::val];

  // Load shared memory
  int t1 = blockIdx.x, t2 = blockIdx.y, t3 = blockIdx.z;
  int c1 = threadIdx.x + Pad<2>::val, c2 = threadIdx.y + Pad<2>::val,
      c3 = threadIdx.z + Pad<2>::val;
  size_t globalOffset =
      (dev_mesh.guard[2] + t3 * DIM3 + c3 - Pad<2>::val) * e1.pitch *
          e1.ysize +
      (dev_mesh.guard[1] + t2 * DIM2 + c2 - Pad<2>::val) * e1.pitch +
      (dev_mesh.guard[0] + t1 * DIM1 + c1 - Pad<2>::val) *
          sizeof(Scalar);

  init_shared_memory<2, DIM1, DIM2, DIM3>(s_e1, s_e2, s_e3, e1, e2, e3,
                                          globalOffset, c1, c2, c3);
  init_shared_memory<2, DIM1, DIM2, DIM3>(s_b1, s_b2, s_b3, b1, b2, b3,
                                          globalOffset, c1, c2, c3);
  __syncthreads();

  Scalar vecE1 = 0.5f * (s_e1[c3][c2][c1] + s_e1[c3][c2][c1 - 1]);
  Scalar vecE2 = 0.5f * (s_e2[c3][c2][c1] + s_e2[c3][c2 - 1][c1]);
  Scalar vecE3 = 0.5f * (s_e3[c3][c2][c1] + s_e3[c3 - 1][c2][c1]);
  Scalar vecB1 =
      0.25f * (s_b1[c3][c2][c1] + s_b1[c3 - 1][c2][c1] +
               s_b1[c3][c2 - 1][c1] + s_b1[c3 - 1][c2 - 1][c1]);
  Scalar vecB2 =
      0.25f * (s_b2[c3][c2][c1] + s_b2[c3 - 1][c2][c1] +
               s_b2[c3][c2][c1 - 1] + s_b2[c3 - 1][c2][c1 - 1]);
  Scalar vecB3 =
      0.25f * (s_b3[c3][c2][c1] + s_b3[c3][c2][c1 - 1] +
               s_b3[c3][c2 - 1][c1] + s_b3[c3][c2 - 1][c1 - 1]);
  Scalar EdotB = vecE1 * vecB1 + vecE2 * vecB2 + vecE3 * vecB3;

  // Do the actual computation here
  (*(Scalar*)((char*)eb.ptr + globalOffset)) += q * EdotB;
}

template <int DIM1, int DIM2, int DIM3>
__global__ void
compute_FFE_J(cudaPitchedPtr j1, cudaPitchedPtr j2, cudaPitchedPtr j3,
              cudaPitchedPtr e1, cudaPitchedPtr e2, cudaPitchedPtr e3,
              cudaPitchedPtr b1, cudaPitchedPtr b2, cudaPitchedPtr b3,
              cudaPitchedPtr f, Scalar q) {
  // Declare cache array in shared memory
  __shared__ Scalar s_e1[DIM3 + 2 * Pad<2>::val][DIM2 + 2 * Pad<2>::val]
                        [DIM1 + 2 * Pad<2>::val];
  __shared__ Scalar s_e2[DIM3 + 2 * Pad<2>::val][DIM2 + 2 * Pad<2>::val]
                        [DIM1 + 2 * Pad<2>::val];
  __shared__ Scalar s_e3[DIM3 + 2 * Pad<2>::val][DIM2 + 2 * Pad<2>::val]
                        [DIM1 + 2 * Pad<2>::val];
  __shared__ Scalar s_b1[DIM3 + 2 * Pad<2>::val][DIM2 + 2 * Pad<2>::val]
                        [DIM1 + 2 * Pad<2>::val];
  __shared__ Scalar s_b2[DIM3 + 2 * Pad<2>::val][DIM2 + 2 * Pad<2>::val]
                        [DIM1 + 2 * Pad<2>::val];
  __shared__ Scalar s_b3[DIM3 + 2 * Pad<2>::val][DIM2 + 2 * Pad<2>::val]
                        [DIM1 + 2 * Pad<2>::val];
  __shared__ Scalar s_f[DIM3 + 2 * Pad<2>::val][DIM2 + 2 * Pad<2>::val]
                       [DIM1 + 2 * Pad<2>::val];

  // Load shared memory
  int t1 = blockIdx.x, t2 = blockIdx.y, t3 = blockIdx.z;
  int c1 = threadIdx.x + Pad<2>::val, c2 = threadIdx.y + Pad<2>::val,
      c3 = threadIdx.z + Pad<2>::val;
  size_t globalOffset =
      (dev_mesh.guard[2] + t3 * DIM3 + c3 - Pad<2>::val) * e1.pitch *
          e1.ysize +
      (dev_mesh.guard[1] + t2 * DIM2 + c2 - Pad<2>::val) * e1.pitch +
      (dev_mesh.guard[0] + t1 * DIM1 + c1 - Pad<2>::val) *
          sizeof(Scalar);

  init_shared_memory<2, DIM1, DIM2, DIM3>(s_e1, s_e2, s_e3, e1, e2, e3,
                                          globalOffset, c1, c2, c3);
  init_shared_memory<2, DIM1, DIM2, DIM3>(s_b1, s_b2, s_b3, b1, b2, b3,
                                          globalOffset, c1, c2, c3);
  init_shared_memory<2, DIM1, DIM2, DIM3>(s_f, f, globalOffset, c1, c2,
                                          c3);
  __syncthreads();

  Scalar vecE1 = 0.5f * (s_e1[c3][c2][c1] + s_e1[c3][c2][c1 - 1]);
  Scalar vecE2 = 0.5f * (s_e2[c3][c2][c1] + s_e2[c3][c2 - 1][c1]);
  Scalar vecE3 = 0.5f * (s_e3[c3][c2][c1] + s_e3[c3 - 1][c2][c1]);
  Scalar vecB1 =
      0.25f * (s_b1[c3][c2][c1] + s_b1[c3 - 1][c2][c1] +
               s_b1[c3][c2 - 1][c1] + s_b1[c3 - 1][c2 - 1][c1]);
  Scalar vecB2 =
      0.25f * (s_b2[c3][c2][c1] + s_b2[c3 - 1][c2][c1] +
               s_b2[c3][c2][c1 - 1] + s_b2[c3 - 1][c2][c1 - 1]);
  Scalar vecB3 =
      0.25f * (s_b3[c3][c2][c1] + s_b3[c3][c2][c1 - 1] +
               s_b3[c3][c2 - 1][c1] + s_b3[c3][c2 - 1][c1 - 1]);
  Scalar inv_B_sqr =
      1.0f / (vecB1 * vecB1 + vecB2 * vecB2 + vecB3 * vecB3);
  Scalar divE =
      (s_e1[c3][c2][c1] - s_e1[c3][c2][c1 - 1]) / dev_mesh.delta[0] +
      (s_e2[c3][c2][c1] - s_e2[c3][c2 - 1][c1]) / dev_mesh.delta[1] +
      (s_e3[c3][c2][c1] - s_e3[c3 - 1][c2][c1]) / dev_mesh.delta[2];
  Scalar EcrossB1 = vecE2 * vecB3 - vecE3 * vecB2;
  Scalar EcrossB2 = vecE3 * vecB1 - vecE1 * vecB3;
  Scalar EcrossB3 = vecE1 * vecB2 - vecE2 * vecB1;
  // Scalar EdotB = vecE1 * vecB1 + vecE2 * vecB2 + vecE3 * vecB3;

  // Do the actual computation here
  (*(Scalar*)((char*)j1.ptr + globalOffset)) =
      q * (s_f[c3][c2][c1] * vecB1 + divE * EcrossB1) * inv_B_sqr;
  (*(Scalar*)((char*)j2.ptr + globalOffset)) =
      q * (s_f[c3][c2][c1] * vecB2 + divE * EcrossB2) * inv_B_sqr;
  (*(Scalar*)((char*)j3.ptr + globalOffset)) =
      q * (s_f[c3][c2][c1] * vecB3 + divE * EcrossB3) * inv_B_sqr;
}

template <int DIM1, int DIM2, int DIM3>
__global__ void
compute_FFE_dE(cudaPitchedPtr e1out, cudaPitchedPtr e2out,
               cudaPitchedPtr e3out, cudaPitchedPtr j1,
               cudaPitchedPtr j2, cudaPitchedPtr j3, cudaPitchedPtr e1,
               cudaPitchedPtr e2, cudaPitchedPtr e3, cudaPitchedPtr b1,
               cudaPitchedPtr b2, cudaPitchedPtr b3, Scalar dt) {
  // Declare cache array in shared memory
  __shared__ Scalar s_e1[DIM3 + 2 * Pad<2>::val][DIM2 + 2 * Pad<2>::val]
                        [DIM1 + 2 * Pad<2>::val];
  __shared__ Scalar s_e2[DIM3 + 2 * Pad<2>::val][DIM2 + 2 * Pad<2>::val]
                        [DIM1 + 2 * Pad<2>::val];
  __shared__ Scalar s_e3[DIM3 + 2 * Pad<2>::val][DIM2 + 2 * Pad<2>::val]
                        [DIM1 + 2 * Pad<2>::val];
  __shared__ Scalar s_b1[DIM3 + 2 * Pad<2>::val][DIM2 + 2 * Pad<2>::val]
                        [DIM1 + 2 * Pad<2>::val];
  __shared__ Scalar s_b2[DIM3 + 2 * Pad<2>::val][DIM2 + 2 * Pad<2>::val]
                        [DIM1 + 2 * Pad<2>::val];
  __shared__ Scalar s_b3[DIM3 + 2 * Pad<2>::val][DIM2 + 2 * Pad<2>::val]
                        [DIM1 + 2 * Pad<2>::val];

  // Load shared memory
  int c1 = threadIdx.x + Pad<2>::val, c2 = threadIdx.y + Pad<2>::val,
      c3 = threadIdx.z + Pad<2>::val;
  size_t globalOffset =
      (dev_mesh.guard[2] + blockIdx.z * DIM3 + c3 - Pad<2>::val) *
          e1.pitch * e1.ysize +
      (dev_mesh.guard[1] + blockIdx.y * DIM2 + c2 - Pad<2>::val) *
          e1.pitch +
      (dev_mesh.guard[0] + blockIdx.x * DIM1 + c1 - Pad<2>::val) *
          sizeof(Scalar);
  init_shared_memory<2, DIM1, DIM2, DIM3>(s_e1, s_e2, s_e3, e1, e2, e3,
                                          globalOffset, c1, c2, c3);
  init_shared_memory<2, DIM1, DIM2, DIM3>(s_b1, s_b2, s_b3, b1, b2, b3,
                                          globalOffset, c1, c2, c3);
  __syncthreads();
  Scalar vecE1 = 0.5f * (s_e1[c3][c2][c1] + s_e1[c3][c2][c1 - 1]);
  Scalar vecE2 = 0.5f * (s_e2[c3][c2][c1] + s_e2[c3][c2 - 1][c1]);
  Scalar vecE3 = 0.5f * (s_e3[c3][c2][c1] + s_e3[c3 - 1][c2][c1]);
  Scalar vecB1 =
      0.25f * (s_b1[c3][c2][c1] + s_b1[c3 - 1][c2][c1] +
               s_b1[c3][c2 - 1][c1] + s_b1[c3 - 1][c2 - 1][c1]);
  Scalar vecB2 =
      0.25f * (s_b2[c3][c2][c1] + s_b2[c3 - 1][c2][c1] +
               s_b2[c3][c2][c1 - 1] + s_b2[c3 - 1][c2][c1 - 1]);
  Scalar vecB3 =
      0.25f * (s_b3[c3][c2][c1] + s_b3[c3][c2][c1 - 1] +
               s_b3[c3][c2 - 1][c1] + s_b3[c3][c2 - 1][c1 - 1]);
  Scalar EcrossB1 = vecE2 * vecB3 - vecE3 * vecB2;
  Scalar EcrossB2 = vecE3 * vecB1 - vecE1 * vecB3;
  Scalar EcrossB3 = vecE1 * vecB2 - vecE2 * vecB1;
  Scalar divE =
      (s_e1[c3][c2][c1] - s_e1[c3][c2][c1 - 1]) / dev_mesh.delta[0] +
      (s_e2[c3][c2][c1] - s_e2[c3][c2 - 1][c1]) / dev_mesh.delta[1] +
      (s_e3[c3][c2][c1] - s_e3[c3 - 1][c2][c1]) / dev_mesh.delta[2];
  Scalar inv_B_sqr =
      1.0f / (vecB1 * vecB1 + vecB2 * vecB2 + vecB3 * vecB3);

  // Compute the second part of the current
  (*(Scalar*)((char*)j1.ptr + globalOffset)) =
      divE * EcrossB1 * inv_B_sqr;
  (*(Scalar*)((char*)j2.ptr + globalOffset)) =
      divE * EcrossB2 * inv_B_sqr;
  (*(Scalar*)((char*)j3.ptr + globalOffset)) =
      divE * EcrossB3 * inv_B_sqr;

  // Reuse EcrossB1, 2, 3 to compute B\dot(curl B)
  EcrossB1 = vecB1 * 0.5f *
             ((s_b3[c3][c2][c1] - s_b3[c3][c2 - 1][c1] +
               s_b3[c3][c2][c1 - 1] - s_b3[c3][c2 - 1][c1 - 1]) *
                  dev_mesh.inv_delta[1] -
              (s_b2[c3][c2][c1] - s_b2[c3 - 1][c2][c1] +
               s_b2[c3][c2][c1 - 1] - s_b2[c3 - 1][c2][c1 - 1]) *
                  dev_mesh.inv_delta[2]);
  EcrossB2 = vecB2 * 0.5f *
             ((s_b1[c3][c2][c1] - s_b1[c3 - 1][c2][c1] +
               s_b1[c3][c2 - 1][c1] - s_b1[c3 - 1][c2 - 1][c1]) *
                  dev_mesh.inv_delta[2] -
              (s_b3[c3][c2][c1] - s_b3[c3][c2][c1 - 1] +
               s_b3[c3][c2 - 1][c1] - s_b3[c3][c2 - 1][c1 - 1]) *
                  dev_mesh.inv_delta[0]);
  EcrossB3 = vecB3 * 0.5f *
             ((s_b2[c3][c2][c1] - s_b2[c3][c2][c1 - 1] +
               s_b2[c3 - 1][c2][c1] - s_b2[c3 - 1][c2][c1 - 1]) *
                  dev_mesh.inv_delta[0] -
              (s_b1[c3][c2][c1] - s_b1[c3][c2 - 1][c1] +
               s_b1[c3 - 1][c2][c1] - s_b1[c3 - 1][c2 - 1][c1]) *
                  dev_mesh.inv_delta[1]);
  // Now use EcrossB1, 2, 3 to compute E\dot(curl E)
  EcrossB1 -= vecE1 * 0.25f *
              ((s_e3[c3][c2 + 1][c1] - s_e3[c3][c2 - 1][c1] +
                s_e3[c3 - 1][c2 + 1][c1] - s_e3[c3 - 1][c2 - 1][c1]) *
                   dev_mesh.inv_delta[1] -
               (s_e2[c3 + 1][c2][c1] - s_e2[c3 - 1][c2][c1] +
                s_e2[c3 + 1][c2 - 1][c1] - s_e2[c3 - 1][c2 - 1][c1]) *
                   dev_mesh.inv_delta[2]);
  EcrossB2 -= vecE2 * 0.25f *
              ((s_e1[c3 + 1][c2][c1] - s_e1[c3 - 1][c2][c1] +
                s_e1[c3 + 1][c2][c1 - 1] - s_e1[c3 - 1][c2][c1 - 1]) *
                   dev_mesh.inv_delta[2] -
               (s_e3[c3][c2][c1 + 1] - s_e3[c3][c2][c1 - 1] +
                s_e3[c3 - 1][c2][c1 + 1] - s_e3[c3 - 1][c2][c1 - 1]) *
                   dev_mesh.inv_delta[0]);
  EcrossB3 -= vecE3 * 0.25f *
              ((s_e2[c3][c2][c1 + 1] - s_e2[c3][c2][c1 - 1] +
                s_e2[c3][c2 - 1][c1 + 1] - s_e2[c3][c2 - 1][c1 - 1]) *
                   dev_mesh.inv_delta[0] -
               (s_e1[c3][c2 + 1][c1] - s_e1[c3][c2 - 1][c1] +
                s_e1[c3][c2 + 1][c1 - 1] - s_e1[c3][c2 - 1][c1 - 1]) *
                   dev_mesh.inv_delta[1]);
  EcrossB1 = EcrossB1 + EcrossB2 + EcrossB3;

  // Compute the first term of the FFE current
  (*(Scalar*)((char*)j1.ptr + globalOffset)) +=
      EcrossB1 * vecB1 * inv_B_sqr;
  (*(Scalar*)((char*)j2.ptr + globalOffset)) +=
      EcrossB1 * vecB2 * inv_B_sqr;
  (*(Scalar*)((char*)j3.ptr + globalOffset)) +=
      EcrossB1 * vecB3 * inv_B_sqr;

  // Now use EcrossB1 to compute curl B
  EcrossB1 =
      (s_b3[c3][c2][c1] - s_b3[c3][c2 - 1][c1]) *
          dev_mesh.inv_delta[1] -
      (s_b2[c3][c2][c1] - s_b2[c3 - 1][c2][c1]) * dev_mesh.inv_delta[2];
  EcrossB2 =
      (s_b1[c3][c2][c1] - s_b1[c3 - 1][c2][c1]) *
          dev_mesh.inv_delta[2] -
      (s_b3[c3][c2][c1] - s_b3[c3][c2][c1 - 1]) * dev_mesh.inv_delta[0];
  EcrossB3 =
      (s_b2[c3][c2][c1] - s_b2[c3][c2][c1 - 1]) *
          dev_mesh.inv_delta[0] -
      (s_b1[c3][c2][c1] - s_b1[c3][c2 - 1][c1]) * dev_mesh.inv_delta[1];

  // Compute the update of E, sans J
  (*(Scalar*)((char*)e1out.ptr + globalOffset)) += dt * EcrossB1;
  (*(Scalar*)((char*)e2out.ptr + globalOffset)) += dt * EcrossB2;
  (*(Scalar*)((char*)e3out.ptr + globalOffset)) += dt * EcrossB3;
}

}  // namespace Kernels

field_solver_ffe::field_solver_ffe(sim_environment& env)
    : m_env(env), m_Etmp(env.grid()), m_Btmp(env.grid()) {}

field_solver_ffe::~field_solver_ffe() {}

void
field_solver_ffe::update_fields(sim_data& data, double dt,
                                double time) {}

void
field_solver_ffe::compute_J(vfield_t& J, vfield_t& E,
                            vfield_t& B) {}

void
field_solver_ffe::update_field_substep(vfield_t& E_out, vfield_t& B_out,
                                       vfield_t& J_out,
                                       vfield_t& E_in,
                                       vfield_t& B_in,
                                       Scalar dt) {
  // Initialize all tmp fields to zero on the device
  // m_tmp.initialize();
  // m_tmp2.initialize();
  m_Etmp.initialize();
  m_Etmp.set_field_type(FieldType::E);

  timer::stamp();
  // Compute the curl of E_in and add it to B_out
  curl_add(B_out, E_in, dt);
  cudaDeviceSynchronize();
  timer::show_duration_since_stamp("First curl and add", "ms");

  // Compute both dE and J together, put the result of J into Etmp
  timer::stamp();
  ffe_dE(E_out, m_Etmp, E_in, B_in, dt);
  cudaDeviceSynchronize();
  timer::show_duration_since_stamp("Computing FFE J", "ms");
  // interpolate J back to staggered position, multiply by dt, and add
  // to E_out
  timer::stamp();
  // m_Etmp.interpolate_from_center_add(E_out, dt);
  cudaDeviceSynchronize();
  timer::show_duration_since_stamp("Interpolate and add", "ms");

  // TODO: Figure out how to best handle removal of the parallel delta_E
}

void
field_solver_ffe::ffe_edotb(scalar_field<Scalar>& result,
                            vector_field<Scalar>& E,
                            vector_field<Scalar>& B,
                            Scalar q) {
  auto& grid = E.grid();
  auto& mesh = grid.mesh();

  dim3 blockSize(8, 8, 8);
  dim3 gridSize(mesh.reduced_dim(0) / 8, mesh.reduced_dim(1) / 8,
                mesh.reduced_dim(2) / 8);
  Kernels::compute_FFE_EdotB<8, 8, 8><<<gridSize, blockSize>>>(
      // result.ptr(), E.ptr(0), E.ptr(1), E.ptr(2), B.ptr(0), B.ptr(1),
      // B.ptr(2),
      get_cudaPitchedPtr(result.data()),
      get_cudaPitchedPtr(E.data(0)),
      get_cudaPitchedPtr(E.data(1)),
      get_cudaPitchedPtr(E.data(2)),
      get_cudaPitchedPtr(B.data(0)),
      get_cudaPitchedPtr(B.data(1)),
      get_cudaPitchedPtr(B.data(2)),
      q);
  CudaCheckError();
}

void
field_solver_ffe::ffe_j(vector_field<Scalar>& result,
                        scalar_field<Scalar>& tmp_f,
                        vector_field<Scalar>& E,
                        vector_field<Scalar>& B, Scalar q) {
  auto& grid = E.grid();
  auto& mesh = grid.mesh();

  dim3 blockSize(8, 8, 4);
  dim3 gridSize(mesh.reduced_dim(0) / 8, mesh.reduced_dim(1) / 8,
                mesh.reduced_dim(2) / 4);

  Kernels::compute_FFE_J<8, 8, 4><<<gridSize, blockSize>>>(
      // result.ptr(0), result.ptr(1), result.ptr(2), E.ptr(0), E.ptr(1),
      // E.ptr(2), B.ptr(0), B.ptr(1), B.ptr(2), tmp_f.ptr()
      get_cudaPitchedPtr(result.data(0)),
      get_cudaPitchedPtr(result.data(1)),
      get_cudaPitchedPtr(result.data(2)),
      get_cudaPitchedPtr(E.data(0)),
      get_cudaPitchedPtr(E.data(1)),
      get_cudaPitchedPtr(E.data(2)),
      get_cudaPitchedPtr(B.data(0)),
      get_cudaPitchedPtr(B.data(1)),
      get_cudaPitchedPtr(B.data(2)),
      get_cudaPitchedPtr(tmp_f.data()), q);
  CudaCheckError();
}

void
field_solver_ffe::ffe_dE(vector_field<Scalar>& Eout,
                         vector_field<Scalar>& J,
                         vector_field<Scalar>& E,
                         vector_field<Scalar>& B, Scalar dt) {
  auto& grid = E.grid();
  auto& mesh = grid.mesh();

  dim3 blockSize(16, 4, 4);
  dim3 gridSize(mesh.reduced_dim(0) / 16, mesh.reduced_dim(1) / 4,
                mesh.reduced_dim(2) / 4);

  Kernels::compute_FFE_dE<16, 4, 4><<<gridSize, blockSize>>>(
      // Eout.ptr(0), Eout.ptr(1), Eout.ptr(2), J.ptr(0), J.ptr(1),
      // J.ptr(2), E.ptr(0), E.ptr(1), E.ptr(2), B.ptr(0), B.ptr(1),
      // B.ptr(2)
      get_cudaPitchedPtr(Eout.data(0)),
      get_cudaPitchedPtr(Eout.data(1)),
      get_cudaPitchedPtr(Eout.data(2)),
      get_cudaPitchedPtr(J.data(0)),
      get_cudaPitchedPtr(J.data(1)),
      get_cudaPitchedPtr(J.data(2)),
      get_cudaPitchedPtr(E.data(0)),
      get_cudaPitchedPtr(E.data(1)),
      get_cudaPitchedPtr(E.data(2)),
      get_cudaPitchedPtr(B.data(0)),
      get_cudaPitchedPtr(B.data(1)),
      get_cudaPitchedPtr(B.data(2)), dt);
  CudaCheckError();
}

}  // namespace Aperture
