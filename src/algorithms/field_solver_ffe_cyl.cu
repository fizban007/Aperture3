#include "algorithms/field_solver_ffe_cyl.h"
#include "algorithms/field_solver_helper.cuh"
#include "algorithms/finite_diff.h"
#include "cuda/constant_mem.h"
#include "cuda/cudaUtility.h"
#include "data/fields_utils.h"
#include "utils/timer.h"

namespace Aperture {

namespace Kernels {

template <int DIM1, int DIM2>
__global__ void
compute_FFE_EdotB(cudaPitchedPtr eb, cudaPitchedPtr e1,
                  cudaPitchedPtr e2, cudaPitchedPtr e3,
                  cudaPitchedPtr b1, cudaPitchedPtr b2,
                  cudaPitchedPtr b3, Scalar q) {
  // Declare cache array in shared memory
  __shared__ Scalar
      s_e1[DIM2 + 2 * Pad<2>::val][DIM1 + 2 * Pad<2>::val];
  __shared__ Scalar
      s_e2[DIM2 + 2 * Pad<2>::val][DIM1 + 2 * Pad<2>::val];
  __shared__ Scalar
      s_e3[DIM2 + 2 * Pad<2>::val][DIM1 + 2 * Pad<2>::val];
  __shared__ Scalar
      s_b1[DIM2 + 2 * Pad<2>::val][DIM1 + 2 * Pad<2>::val];
  __shared__ Scalar
      s_b2[DIM2 + 2 * Pad<2>::val][DIM1 + 2 * Pad<2>::val];
  __shared__ Scalar
      s_b3[DIM2 + 2 * Pad<2>::val][DIM1 + 2 * Pad<2>::val];

  // Load shared memory
  int t1 = blockIdx.x, t2 = blockIdx.y;
  int c1 = threadIdx.x + Pad<2>::val, c2 = threadIdx.y + Pad<2>::val;
  size_t globalOffset =
      (dev_mesh.guard[1] + t2 * DIM2 + c2 - Pad<2>::val) * e1.pitch +
      (dev_mesh.guard[0] + t1 * DIM1 + c1 - Pad<2>::val) *
          sizeof(Scalar);

  init_shared_memory_2d<2, DIM1, DIM2>(s_e1, s_e2, s_e3, e1, e2, e3,
                                       globalOffset, c1, c2);
  init_shared_memory_2d<2, DIM1, DIM2>(s_b1, s_b2, s_b3, b1, b2, b3,
                                       globalOffset, c1, c2);
  __syncthreads();

  Scalar vecE1 = 0.5f * (s_e1[c2][c1] + s_e1[c2][c1 - 1]);
  Scalar vecE2 = 0.5f * (s_e2[c2][c1] + s_e2[c2 - 1][c1]);
  Scalar vecE3 = s_e3[c2][c1];
  Scalar vecB1 = 0.5f * (s_b1[c2][c1] + s_b1[c2 - 1][c1]);
  Scalar vecB2 = 0.5f * (s_b2[c2][c1] + s_b2[c2][c1 - 1]);
  Scalar vecB3 = 0.25f * (s_b3[c2][c1] + s_b3[c2][c1 - 1] +
                          s_b3[c2 - 1][c1] + s_b3[c2 - 1][c1 - 1]);
  Scalar EdotB = vecE1 * vecB1 + vecE2 * vecB2 + vecE3 * vecB3;

  // Do the actual computation here
  (*(Scalar*)((char*)eb.ptr + globalOffset)) += q * EdotB;
}

template <int DIM1, int DIM2>
__global__ void
compute_FFE_J(cudaPitchedPtr j1, cudaPitchedPtr j2, cudaPitchedPtr j3,
              cudaPitchedPtr e1, cudaPitchedPtr e2, cudaPitchedPtr e3,
              cudaPitchedPtr b1, cudaPitchedPtr b2, cudaPitchedPtr b3,
              cudaPitchedPtr f, Scalar q) {
  // Declare cache array in shared memory
  __shared__ Scalar
      s_e1[DIM2 + 2 * Pad<2>::val][DIM1 + 2 * Pad<2>::val];
  __shared__ Scalar
      s_e2[DIM2 + 2 * Pad<2>::val][DIM1 + 2 * Pad<2>::val];
  __shared__ Scalar
      s_e3[DIM2 + 2 * Pad<2>::val][DIM1 + 2 * Pad<2>::val];
  __shared__ Scalar
      s_b1[DIM2 + 2 * Pad<2>::val][DIM1 + 2 * Pad<2>::val];
  __shared__ Scalar
      s_b2[DIM2 + 2 * Pad<2>::val][DIM1 + 2 * Pad<2>::val];
  __shared__ Scalar
      s_b3[DIM2 + 2 * Pad<2>::val][DIM1 + 2 * Pad<2>::val];
  __shared__ Scalar s_f[DIM2 + 2 * Pad<2>::val][DIM1 + 2 * Pad<2>::val];

  // Load shared memory
  int t1 = blockIdx.x, t2 = blockIdx.y;
  int c1 = threadIdx.x + Pad<2>::val, c2 = threadIdx.y + Pad<2>::val;
  size_t globalOffset =
      (dev_mesh.guard[1] + t2 * DIM2 + c2 - Pad<2>::val) * e1.pitch +
      (dev_mesh.guard[0] + t1 * DIM1 + c1 - Pad<2>::val) *
          sizeof(Scalar);

  init_shared_memory_2d<2, DIM1, DIM2>(s_e1, s_e2, s_e3, e1, e2, e3,
                                       globalOffset, c1, c2);
  init_shared_memory_2d<2, DIM1, DIM2>(s_b1, s_b2, s_b3, b1, b2, b3,
                                       globalOffset, c1, c2);
  init_shared_memory_2d<2, DIM1, DIM2>(s_f, f, globalOffset, c1, c2);
  __syncthreads();

  Scalar vecE1 = 0.5f * (s_e1[c2][c1] + s_e1[c2][c1 - 1]);
  Scalar vecE2 = 0.5f * (s_e2[c2][c1] + s_e2[c2 - 1][c1]);
  Scalar vecE3 = s_e3[c2][c1];
  Scalar vecB1 = 0.5f * (s_b1[c2][c1] + s_b1[c2 - 1][c1]);
  Scalar vecB2 = 0.5f * (s_b2[c2][c1] + s_b2[c2][c1 - 1]);
  Scalar vecB3 = 0.25f * (s_b3[c2][c1] + s_b3[c2][c1 - 1] +
                          s_b3[c2 - 1][c1] + s_b3[c2 - 1][c1 - 1]);
  Scalar inv_B_sqr =
      1.0f / (vecB1 * vecB1 + vecB2 * vecB2 + vecB3 * vecB3);
  Scalar r1 = dev_mesh.pos(
      0, dev_mesh.guard[0] + t1 * DIM1 + threadIdx.x, false);
  Scalar r0 = dev_mesh.pos(
      0, dev_mesh.guard[0] + t1 * DIM1 + threadIdx.x - 1, false);
  Scalar divE = (r1 * s_e1[c2][c1] - r0 * s_e1[c2][c1 - 1]) /
                    (0.5f * (r0 + r1) * dev_mesh.delta[0]) +
                (s_e2[c2][c1] - s_e2[c2 - 1][c1]) / dev_mesh.delta[1];
  Scalar EcrossB1 = vecE2 * vecB3 - vecE3 * vecB2;
  Scalar EcrossB2 = vecE3 * vecB1 - vecE1 * vecB3;
  Scalar EcrossB3 = vecE1 * vecB2 - vecE2 * vecB1;
  // Scalar EdotB = vecE1 * vecB1 + vecE2 * vecB2 + vecE3 * vecB3;

  // Do the actual computation here
  (*(Scalar*)((char*)j1.ptr + globalOffset)) =
      q * (s_f[c2][c1] * vecB1 + divE * EcrossB1) * inv_B_sqr;
  (*(Scalar*)((char*)j2.ptr + globalOffset)) =
      q * (s_f[c2][c1] * vecB2 + divE * EcrossB2) * inv_B_sqr;
  (*(Scalar*)((char*)j3.ptr + globalOffset)) =
      q * (s_f[c2][c1] * vecB3 + divE * EcrossB3) * inv_B_sqr;
}

template <int DIM1, int DIM2>
__global__ void
compute_FFE_dE(cudaPitchedPtr e1out, cudaPitchedPtr e2out,
               cudaPitchedPtr e3out, cudaPitchedPtr j1,
               cudaPitchedPtr j2, cudaPitchedPtr j3, cudaPitchedPtr e1,
               cudaPitchedPtr e2, cudaPitchedPtr e3, cudaPitchedPtr b1,
               cudaPitchedPtr b2, cudaPitchedPtr b3, Scalar dt) {
  // Declare cache array in shared memory
  __shared__ Scalar
      s_e1[DIM2 + 2 * Pad<2>::val][DIM1 + 2 * Pad<2>::val];
  __shared__ Scalar
      s_e2[DIM2 + 2 * Pad<2>::val][DIM1 + 2 * Pad<2>::val];
  __shared__ Scalar
      s_e3[DIM2 + 2 * Pad<2>::val][DIM1 + 2 * Pad<2>::val];
  __shared__ Scalar
      s_b1[DIM2 + 2 * Pad<2>::val][DIM1 + 2 * Pad<2>::val];
  __shared__ Scalar
      s_b2[DIM2 + 2 * Pad<2>::val][DIM1 + 2 * Pad<2>::val];
  __shared__ Scalar
      s_b3[DIM2 + 2 * Pad<2>::val][DIM1 + 2 * Pad<2>::val];

  // Load shared memory
  int c1 = threadIdx.x + Pad<2>::val, c2 = threadIdx.y + Pad<2>::val;
  size_t globalOffset =
      (dev_mesh.guard[1] + blockIdx.y * DIM2 + c2 - Pad<2>::val) *
          e1.pitch +
      (dev_mesh.guard[0] + blockIdx.x * DIM1 + c1 - Pad<2>::val) *
          sizeof(Scalar);
  init_shared_memory_2d<2, DIM1, DIM2>(s_e1, s_e2, s_e3, e1, e2, e3,
                                       globalOffset, c1, c2);
  init_shared_memory_2d<2, DIM1, DIM2>(s_b1, s_b2, s_b3, b1, b2, b3,
                                       globalOffset, c1, c2);
  __syncthreads();
  Scalar vecE1 = 0.5f * (s_e1[c2][c1] + s_e1[c2][c1 - 1]);
  Scalar vecE2 = 0.5f * (s_e2[c2][c1] + s_e2[c2 - 1][c1]);
  Scalar vecE3 = s_e3[c2][c1];
  Scalar vecB1 = 0.5f * (s_b1[c2][c1] + s_b1[c2 - 1][c1]);
  Scalar vecB2 = 0.5f * (s_b2[c2][c1] + s_b2[c2][c1 - 1]);
  Scalar vecB3 = 0.25f * (s_b3[c2][c1] + s_b3[c2][c1 - 1] +
                          s_b3[c2 - 1][c1] + s_b3[c2 - 1][c1 - 1]);
  Scalar EcrossB1 = vecE2 * vecB3 - vecE3 * vecB2;
  Scalar EcrossB2 = vecE3 * vecB1 - vecE1 * vecB3;
  Scalar EcrossB3 = vecE1 * vecB2 - vecE2 * vecB1;
  Scalar r1 = dev_mesh.pos(
      0, dev_mesh.guard[0] + blockIdx.x * DIM1 + threadIdx.x, false);
  Scalar r0 = dev_mesh.pos(
      0, dev_mesh.guard[0] + blockIdx.x * DIM1 + threadIdx.x - 1,
      false);
  Scalar r1s = dev_mesh.pos(
      0, dev_mesh.guard[0] + blockIdx.x * DIM1 + threadIdx.x, true);
  Scalar r0s = dev_mesh.pos(
      0, dev_mesh.guard[0] + blockIdx.x * DIM1 + threadIdx.x - 1, true);
  Scalar divE =
      (r1s * s_e1[c2][c1] - r0s * s_e1[c2][c1 - 1]) /
          (0.5f * (r0s + r1s) * dev_mesh.delta[0]) +
      (s_e2[c2][c1] - s_e2[c2 - 1][c1]) * dev_mesh.inv_delta[1];
  Scalar inv_B_sqr =
      1.0f / (vecB1 * vecB1 + vecB2 * vecB2 + vecB3 * vecB3);

  // Compute the second part of the current
  (*(Scalar*)((char*)j1.ptr + globalOffset)) =
      divE * EcrossB1 * inv_B_sqr * dt;
  (*(Scalar*)((char*)j2.ptr + globalOffset)) =
      divE * EcrossB2 * inv_B_sqr * dt;
  (*(Scalar*)((char*)j3.ptr + globalOffset)) =
      divE * EcrossB3 * inv_B_sqr * dt;

  // Reuse EcrossB1, 2, 3 to compute B\dot(curl B)
  // TODO: Add cylindrical coefficients
  EcrossB1 = vecB1 * 0.5f *
             (s_b3[c2][c1] - s_b3[c2 - 1][c1] + s_b3[c2][c1 - 1] -
              s_b3[c2 - 1][c1 - 1]) *
             dev_mesh.inv_delta[1];
  EcrossB2 = vecB2 * 0.5f *
             (r1s * s_b3[c2][c1] - r0s * s_b3[c2][c1 - 1] +
              r1s * s_b3[c2 - 1][c1] - r0s * s_b3[c2 - 1][c1 - 1]) *
             dev_mesh.inv_delta[0] * 2.0f / (r1s + r0s);
  EcrossB3 =
      vecB3 *
      ((s_b2[c2][c1] - s_b2[c2][c1 - 1]) * dev_mesh.inv_delta[0] -
       (s_b1[c2][c1] - s_b1[c2 - 1][c1]) * dev_mesh.inv_delta[1]);
  // Now use EcrossB1, 2, 3 to compute E\dot(curl E)
  // TODO: Add cylindrical coefficients
  EcrossB1 -= vecE1 * 0.5f * (s_e3[c2 + 1][c1] - s_e3[c2 - 1][c1]) *
              dev_mesh.inv_delta[1];
  EcrossB2 -= vecE2 * 0.5f *
              ((r1 + dev_mesh.delta[0]) * s_e3[c2][c1 + 1] -
               r0 * s_e3[c2][c1 - 1]) *
              dev_mesh.inv_delta[0] / r1;
  EcrossB3 -= vecE3 * 0.25f *
              ((s_e2[c2][c1 + 1] - s_e2[c2][c1 - 1] +
                s_e2[c2 - 1][c1 + 1] - s_e2[c2 - 1][c1 - 1]) *
                   dev_mesh.inv_delta[0] -
               (s_e1[c2 + 1][c1] - s_e1[c2 - 1][c1] +
                s_e1[c2 + 1][c1 - 1] - s_e1[c2 - 1][c1 - 1]) *
                   dev_mesh.inv_delta[1]);
  EcrossB1 = EcrossB1 + EcrossB2 + EcrossB3;

  // Compute the first term of the FFE current
  (*(Scalar*)((char*)j1.ptr + globalOffset)) +=
      EcrossB1 * vecB1 * inv_B_sqr * dt;
  (*(Scalar*)((char*)j2.ptr + globalOffset)) +=
      EcrossB1 * vecB2 * inv_B_sqr * dt;
  (*(Scalar*)((char*)j3.ptr + globalOffset)) +=
      EcrossB1 * vecB3 * inv_B_sqr * dt;

  // Now use EcrossB1 to compute curl B
  EcrossB1 = (s_b3[c2][c1] - s_b3[c2 - 1][c1]) * dev_mesh.inv_delta[1];
  EcrossB2 = (r1s * s_b3[c2][c1] - r0s * s_b3[c2][c1 - 1]) *
             dev_mesh.inv_delta[0] * 2.0f / (r1s + r0s);
  EcrossB3 = (s_b2[c2][c1] - s_b2[c2][c1 - 1]) * dev_mesh.inv_delta[0] -
             (s_b1[c2][c1] - s_b1[c2 - 1][c1]) * dev_mesh.inv_delta[1];

  // Compute the update of E, sans J
  (*(Scalar*)((char*)e1out.ptr + globalOffset)) = s_e1[c2][c1] + dt * EcrossB1;
  (*(Scalar*)((char*)e2out.ptr + globalOffset)) = s_e2[c2][c1] + dt * EcrossB2;
  (*(Scalar*)((char*)e3out.ptr + globalOffset)) = s_e3[c2][c1] + dt * EcrossB3;
}

template <int DIM1, int DIM2>
__global__ void
compute_curl_add_2d_cyl(cudaPitchedPtr v1, cudaPitchedPtr v2,
                        cudaPitchedPtr v3, cudaPitchedPtr u1,
                        cudaPitchedPtr u2, cudaPitchedPtr u3,
                        Stagger s1, Stagger s2, Stagger s3,
                        Scalar q = 1.0) {
  // Declare cache array in shared memory
  __shared__ Scalar
      s_u1[DIM2 + 2 * Pad<2>::val][DIM1 + 2 * Pad<2>::val];
  __shared__ Scalar
      s_u2[DIM2 + 2 * Pad<2>::val][DIM1 + 2 * Pad<2>::val];
  __shared__ Scalar
      s_u3[DIM2 + 2 * Pad<2>::val][DIM1 + 2 * Pad<2>::val];

  // Load shared memory
  int t1 = blockIdx.x, t2 = blockIdx.y;
  int c1 = threadIdx.x + Pad<2>::val, c2 = threadIdx.y + Pad<2>::val;
  size_t globalOffset =
      (dev_mesh.guard[1] + t2 * DIM2 + c2 - Pad<2>::val) * u1.pitch +
      (dev_mesh.guard[0] + t1 * DIM1 + c1 - Pad<2>::val) *
          sizeof(Scalar);

  init_shared_memory_2d<2, DIM1, DIM2>(s_u1, s_u2, s_u3, u1, u2, u3,
                                       globalOffset, c1, c2);
  __syncthreads();

  // Do the actual computation here
  // (Curl u)_1 = d2u3 - d3u2
  (*(Scalar*)((char*)v1.ptr + globalOffset)) +=
      q *
      (s_u3[c2 + flip(s3[1])][c1] - s_u3[c2 - 1 + flip(s3[1])][c1]) *
      dev_mesh.inv_delta[1];

  // (Curl u)_2 = d3u1 - d1u3
  Scalar r1 = dev_mesh.pos(
      0, dev_mesh.guard[0] + t1 * DIM1 + threadIdx.x, flip(s3[0]));
  Scalar r0 = dev_mesh.pos(
      0, dev_mesh.guard[0] + t1 * DIM1 + threadIdx.x - 1, flip(s3[0]));
  (*(Scalar*)((char*)v2.ptr + globalOffset)) +=
      q *
      (r1 * s_u3[c2][c1 + flip(s3[0])] -
       r0 * s_u3[c2][c1 - 1 + flip(s3[0])]) *
      dev_mesh.inv_delta[0] * 2.0f / (r1 + r0);

  // (Curl u)_3 = d1u2 - d2u1
  (*(Scalar*)((char*)v3.ptr + globalOffset)) +=
      q *
      ((s_u2[c2][c1 + flip(s2[0])] - s_u2[c2][c1 - 1 + flip(s2[0])]) *
           dev_mesh.inv_delta[0] -
       (s_u1[c2 + flip(s1[1])][c1] - s_u1[c2 - 1 + flip(s1[1])][c1]) *
           dev_mesh.inv_delta[1]);
}

template <int DIM1, int DIM2>
__global__ void
reduce_E(cudaPitchedPtr e1, cudaPitchedPtr e2, cudaPitchedPtr e3,
         cudaPitchedPtr b1, cudaPitchedPtr b2, cudaPitchedPtr b3,
         cudaPitchedPtr ec1, cudaPitchedPtr ec2, cudaPitchedPtr ec3) {
  // Declare cache array in shared memory
  __shared__ Scalar
      s_e1[DIM2 + 2 * Pad<2>::val][DIM1 + 2 * Pad<2>::val];
  __shared__ Scalar
      s_e2[DIM2 + 2 * Pad<2>::val][DIM1 + 2 * Pad<2>::val];
  __shared__ Scalar
      s_e3[DIM2 + 2 * Pad<2>::val][DIM1 + 2 * Pad<2>::val];
  __shared__ Scalar
      s_b1[DIM2 + 2 * Pad<2>::val][DIM1 + 2 * Pad<2>::val];
  __shared__ Scalar
      s_b2[DIM2 + 2 * Pad<2>::val][DIM1 + 2 * Pad<2>::val];
  __shared__ Scalar
      s_b3[DIM2 + 2 * Pad<2>::val][DIM1 + 2 * Pad<2>::val];

  // Load shared memory
  int c1 = threadIdx.x + Pad<2>::val, c2 = threadIdx.y + Pad<2>::val;
  size_t globalOffset =
      (dev_mesh.guard[1] + blockIdx.y * DIM2 + c2 - Pad<2>::val) *
          e1.pitch +
      (dev_mesh.guard[0] + blockIdx.x * DIM1 + c1 - Pad<2>::val) *
          sizeof(Scalar);
  init_shared_memory_2d<2, DIM1, DIM2>(s_e1, s_e2, s_e3, e1, e2, e3,
                                       globalOffset, c1, c2);
  init_shared_memory_2d<2, DIM1, DIM2>(s_b1, s_b2, s_b3, b1, b2, b3,
                                       globalOffset, c1, c2);
  __syncthreads();

  // Remove EdotB component
  Scalar vecE1 = 0.5f * (s_e1[c2][c1] + s_e1[c2][c1 - 1]) +
                 *(Scalar*)((char*)ec1.ptr + globalOffset);
  Scalar vecE2 = 0.5f * (s_e2[c2][c1] + s_e2[c2 - 1][c1]) +
                 *(Scalar*)((char*)ec2.ptr + globalOffset);
  Scalar vecE3 =
      s_e3[c2][c1] + *(Scalar*)((char*)ec3.ptr + globalOffset);
  Scalar vecB1 = 0.5f * (s_b1[c2][c1] + s_b1[c2 - 1][c1]);
  Scalar vecB2 = 0.5f * (s_b2[c2][c1] + s_b2[c2][c1 - 1]);
  Scalar vecB3 = 0.25f * (s_b3[c2][c1] + s_b3[c2][c1 - 1] +
                          s_b3[c2 - 1][c1] + s_b3[c2 - 1][c1 - 1]);
  Scalar B_sqr = vecB1 * vecB1 + vecB2 * vecB2 + vecB3 * vecB3;
  Scalar EdotB = vecE1 * vecB1 + vecE2 * vecB2 + vecE3 * vecB3;

  if (std::abs(EdotB) > EPS) {
    vecE1 -= EdotB * vecB1 / B_sqr;
    vecE2 -= EdotB * vecB2 / B_sqr;
    vecE3 -= EdotB * vecB3 / B_sqr;
  }

  // Reduce E if E is still larger than B
  Scalar E_sqr = vecE1 * vecE1 + vecE2 * vecE2 + vecE3 * vecE3;
  if (E_sqr > B_sqr) {
    vecE1 *= std::sqrt(B_sqr / E_sqr);
    vecE2 *= std::sqrt(B_sqr / E_sqr);
    vecE3 *= std::sqrt(B_sqr / E_sqr);
  }

  __syncthreads();
  (*(Scalar*)((char*)ec1.ptr + globalOffset)) = vecE1;
  (*(Scalar*)((char*)ec2.ptr + globalOffset)) = vecE2;
  (*(Scalar*)((char*)ec3.ptr + globalOffset)) = vecE3;
}

}  // namespace Kernels

void
curl_add_2d(VectorField<Scalar>& result, const VectorField<Scalar>& u,
            Scalar q) {
  auto& mesh = u.grid().mesh();

  dim3 blockSize(32, 16);
  dim3 gridSize(mesh.reduced_dim(0) / 32, mesh.reduced_dim(1) / 16);
  Kernels::compute_curl_add_2d_cyl<32, 16><<<gridSize, blockSize>>>(
      result.ptr(0), result.ptr(1), result.ptr(2), u.ptr(0), u.ptr(1),
      u.ptr(2), u.stagger(0), u.stagger(1), u.stagger(2), q);
  CudaCheckError();
}

FieldSolver_FFE_Cyl::FieldSolver_FFE_Cyl(const Grid& g)
    : m_Etmp(g),
        m_Etmp2(g),
      m_Erk(g),
      m_Brk(g)
// , m_tmp2(g),
// m_e1(g), m_e2(g), m_e3(g), m_e4(g),
// m_b1(g), m_b2(g), m_b3(g), m_b4(g)
{
  m_Brk.set_field_type(FieldType::B);
  // m_j1(g), m_j2(g), m_j3(g), m_j4(g) {
  // m_b1.set_field_type(FieldType::B);
  // m_b2.set_field_type(FieldType::B);
  // m_b3.set_field_type(FieldType::B);
  // m_b4.set_field_type(FieldType::B);
  m_a[0] = 0.0f;
  m_a[1] = 1.0f / 3.0f;
  m_a[2] = 1.0f;
  m_a[3] = 1.0f;

  m_b[0] = 1.0f / 8.0f;
  m_b[1] = 3.0f / 8.0f;
  m_b[2] = 3.0f / 8.0f;
  m_b[3] = 1.0f / 8.0f;

  m_c[0] = 0.0f;
  m_c[1] = 1.0f / 3.0f;
  m_c[2] = 2.0f / 3.0f;
  m_c[3] = 1.0f;
}

FieldSolver_FFE_Cyl::~FieldSolver_FFE_Cyl() {}

void
FieldSolver_FFE_Cyl::update_fields(SimData& data, double dt,
                                   double time) {
  // Apply Low Storage RK4 method here:
  for (int n = 0; n < 4; n++) {
    // timer::stamp();
    if (n > 0) {
      // m_Erk = data.E;
      // m_Brk = data.B;
      // m_Erk.multiplyBy(m_a[n] + m_b[n - 1]);
      // m_Brk.multiplyBy(m_a[n] + m_b[n - 1]);
      m_Erk.assign(data.E, m_a[n] - m_b[n - 1]);
      m_Brk.assign(data.B, m_a[n] - m_b[n - 1]);
    }
    update_field_substep(data.E, data.B, data.J,
                         m_Erk, m_Brk, m_b[n] * dt);
    // data.E.addBy(m_Erk, m_b[n] * dt);
    // data.B.addBy(m_Brk, m_b[n] * dt);
    // timer::show_duration_since_stamp("FFE substep", "ms");
  }
}

void
FieldSolver_FFE_Cyl::compute_J(vfield_t& J, const vfield_t& E,
                               const vfield_t& B) {}

void
FieldSolver_FFE_Cyl::update_field_substep(
    vfield_t& E_out, vfield_t& B_out, vfield_t& J_out,
    const vfield_t& E_in, const vfield_t& B_in, Scalar dt) {
  // Initialize all tmp fields to zero on the device
  // m_tmp.initialize();
  // m_tmp2.initialize();
  m_Etmp.initialize();
  m_Etmp.set_field_type(FieldType::E);
  // m_Etmp2.copyFrom(E_in);
  // m_Etmp2.set_field_type(FieldType::E);

  // timer::stamp();
  // Compute the curl of E_in and add it to B_out
  curl_add_2d(B_out, E_in, dt);
  // cudaDeviceSynchronize();
  // timer::show_duration_since_stamp("First curl and add", "ms");

  // Compute both dE and J together, put the result of J into Etmp, and
  // update m_Etmp2 with the curl of B_in
  // timer::stamp();
  ffe_dE(m_Etmp2, m_Etmp, E_in, B_in, dt);
  cudaDeviceSynchronize();

  // Interpolate m_Etmp back to J_out, removing the dt factor
  m_Etmp.interpolate_from_center(J_out, 1.0f / dt);
  cudaDeviceSynchronize();
  // timer::show_duration_since_stamp("Computing FFE J", "ms");

  // Handle removal of the parallel delta_E, and when E larger than B
  // timer::stamp();
  ffe_reduceE(m_Etmp, m_Etmp2, B_out);
  cudaDeviceSynchronize();
  // timer::show_duration_since_stamp("Reducing FFE E", "ms");

  // Interpolate the result from the center to E_out
  // timer::stamp();
  m_Etmp.interpolate_from_center(E_out);
  cudaDeviceSynchronize();
  // timer::show_duration_since_stamp("Interpolate and add", "ms");
}

void
FieldSolver_FFE_Cyl::ffe_edotb(ScalarField<Scalar>& result,
                               const VectorField<Scalar>& E,
                               const VectorField<Scalar>& B, Scalar q) {
  auto& grid = E.grid();
  auto& mesh = grid.mesh();

  dim3 blockSize(32, 16);
  dim3 gridSize(mesh.reduced_dim(0) / 32, mesh.reduced_dim(1) / 16);
  Kernels::compute_FFE_EdotB<32, 16><<<gridSize, blockSize>>>(
      result.ptr(), E.ptr(0), E.ptr(1), E.ptr(2), B.ptr(0), B.ptr(1),
      B.ptr(2), q);
  CudaCheckError();
}

void
FieldSolver_FFE_Cyl::ffe_j(VectorField<Scalar>& result,
                           const ScalarField<Scalar>& tmp_f,
                           const VectorField<Scalar>& E,
                           const VectorField<Scalar>& B, Scalar q) {
  auto& grid = E.grid();
  auto& mesh = grid.mesh();

  dim3 blockSize(32, 16);
  dim3 gridSize(mesh.reduced_dim(0) / 32, mesh.reduced_dim(1) / 16);

  Kernels::compute_FFE_J<32, 16><<<gridSize, blockSize>>>(
      result.ptr(0), result.ptr(1), result.ptr(2), E.ptr(0), E.ptr(1),
      E.ptr(2), B.ptr(0), B.ptr(1), B.ptr(2), tmp_f.ptr(), q);
  CudaCheckError();
}

void
FieldSolver_FFE_Cyl::ffe_dE(VectorField<Scalar>& Eout,
                            VectorField<Scalar>& J,
                            const VectorField<Scalar>& E,
                            const VectorField<Scalar>& B, Scalar dt) {
  auto& grid = E.grid();
  auto& mesh = grid.mesh();

  dim3 blockSize(32, 16);
  dim3 gridSize(mesh.reduced_dim(0) / 32, mesh.reduced_dim(1) / 16);

  Kernels::compute_FFE_dE<32, 16><<<gridSize, blockSize>>>(
      Eout.ptr(0), Eout.ptr(1), Eout.ptr(2), J.ptr(0), J.ptr(1),
      J.ptr(2), E.ptr(0), E.ptr(1), E.ptr(2), B.ptr(0), B.ptr(1),
      B.ptr(2), dt);
  CudaCheckError();
}

void
FieldSolver_FFE_Cyl::ffe_reduceE(VectorField<Scalar>& E_center,
                                 const VectorField<Scalar>& E,
                                 const VectorField<Scalar>& B) {
  auto& grid = E.grid();
  auto& mesh = grid.mesh();

  dim3 blockSize(32, 16);
  dim3 gridSize(mesh.reduced_dim(0) / 32, mesh.reduced_dim(1) / 16);

  Kernels::reduce_E<32, 16><<<gridSize, blockSize>>>(
      E.ptr(0), E.ptr(1), E.ptr(2), B.ptr(0), B.ptr(1), B.ptr(2),
      E_center.ptr(0), E_center.ptr(1), E_center.ptr(2));
  CudaCheckError();
}

}  // namespace Aperture