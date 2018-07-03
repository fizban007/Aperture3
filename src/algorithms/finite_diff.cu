#include "algorithms/finite_diff.h"
#include "data/stagger.h"
#include "cuda/constant_mem.h"
#include "cuda/cudaUtility.h"
#include "utils/util_functions.h"

namespace Aperture {

namespace Kernels {

HD_INLINE
Scalar deriv(Scalar f0, Scalar f1, Scalar delta) {
  return (f1 - f0) / delta;
}

template <int DIM1, int DIM2, int DIM3>
__device__ __forceinline__
Scalar d1(Scalar array[][DIM2 + 2][DIM1 + 2], int c1, int c2, int c3) {
  // printf("d1, %f, %f, %f, %f\n", array[c3][c2][c1 - 1], array[c3][c2][c1], dev_mesh.delta[0],
  //        deriv(array[c3][c2][c1 - 1], array[c3][c2][c1], dev_mesh.delta[0]));
  return deriv(array[c3][c2][c1 - 1], array[c3][c2][c1], dev_mesh.delta[0]);
}

template <int DIM1, int DIM2, int DIM3>
__device__ __forceinline__
Scalar d2(Scalar array[][DIM2 + 2][DIM1 + 2], int c1, int c2, int c3) {
  // printf("d2, %f, %f, %f, %f\n", array[c3][c2 - 1][c1], array[c3][c2][c1], dev_mesh.delta[1],
  //        deriv(array[c3][c2 - 1][c1], array[c3][c2][c1], dev_mesh.delta[1]));
  return deriv(array[c3][c2 - 1][c1], array[c3][c2][c1], dev_mesh.delta[1]);
}

template <int DIM1, int DIM2, int DIM3>
__device__ __forceinline__
Scalar d3(Scalar array[][DIM2 + 2][DIM1 + 2], int c1, int c2, int c3) {
  return deriv(array[c3 - 1][c2][c1], array[c3][c2][c1], dev_mesh.delta[2]);
}

template <int DIM1, int DIM2, int DIM3>
__device__ __forceinline__
void init_shared_memory(Scalar s_u1[][DIM2 + 2][DIM1 + 2],
                        Scalar s_u2[][DIM2 + 2][DIM1 + 2],
                        Scalar s_u3[][DIM2 + 2][DIM1 + 2],
                        const Scalar* u1, const Scalar* u2, const Scalar* u3,
                        int globalIdx, int c1, int c2, int c3) {
  // Load field values into shared memory
  s_u1[c3][c2][c1] = u1[globalIdx];
  s_u2[c3][c2][c1] = u2[globalIdx];
  s_u3[c3][c2][c1] = u3[globalIdx];

  // Handle extra guard cells
  if (c1 == 1) {
    s_u1[c3][c2][c1 - 1] = u1[globalIdx - 1];
    s_u2[c3][c2][c1 - 1] = u2[globalIdx - 1];
    s_u3[c3][c2][c1 - 1] = u3[globalIdx - 1];
  } else if (c1 == DIM1) {
    s_u1[c3][c2][c1 + 1] = u1[globalIdx + 1];
    s_u2[c3][c2][c1 + 1] = u2[globalIdx + 1];
    s_u3[c3][c2][c1 + 1] = u3[globalIdx + 1];
  }
  if (c2 == 1) {
    s_u1[c3][c2 - 1][c1] = u1[globalIdx - dev_mesh.dims[0]];
    s_u2[c3][c2 - 1][c1] = u2[globalIdx - dev_mesh.dims[0]];
    s_u3[c3][c2 - 1][c1] = u3[globalIdx - dev_mesh.dims[0]];
  } else if (c2 == DIM2) {
    s_u1[c3][c2 + 1][c1] = u1[globalIdx + dev_mesh.dims[0]];
    s_u2[c3][c2 + 1][c1] = u2[globalIdx + dev_mesh.dims[0]];
    s_u3[c3][c2 + 1][c1] = u3[globalIdx + dev_mesh.dims[0]];
  }
  if (c3 == 1) {
    s_u1[c3 - 1][c2][c1] = u1[globalIdx - dev_mesh.dims[0] * dev_mesh.dims[1]];
    s_u2[c3 - 1][c2][c1] = u2[globalIdx - dev_mesh.dims[0] * dev_mesh.dims[1]];
    s_u3[c3 - 1][c2][c1] = u3[globalIdx - dev_mesh.dims[0] * dev_mesh.dims[1]];
  } else if (c3 == DIM3) {
    s_u1[c3 + 1][c2][c1] = u1[globalIdx + dev_mesh.dims[0] * dev_mesh.dims[1]];
    s_u2[c3 + 1][c2][c1] = u2[globalIdx + dev_mesh.dims[0] * dev_mesh.dims[1]];
    s_u3[c3 + 1][c2][c1] = u3[globalIdx + dev_mesh.dims[0] * dev_mesh.dims[1]];
  }
}

template <int DIM1, int DIM2, int DIM3>
__global__
void compute_curl(Scalar* v1, Scalar* v2, Scalar* v3,
                  const Scalar* u1, const Scalar* u2, const Scalar* u3,
                  Stagger s1, Stagger s2, Stagger s3) {
  // Declare cache array in shared memory
  __shared__ Scalar s_u1[DIM3 + 2][DIM2 + 2][DIM1 + 2];
  __shared__ Scalar s_u2[DIM3 + 2][DIM2 + 2][DIM1 + 2];
  __shared__ Scalar s_u3[DIM3 + 2][DIM2 + 2][DIM1 + 2];

  // Load indices
  int t1 = blockIdx.x, t2 = blockIdx.y, t3 = blockIdx.z;
  int c1 = threadIdx.x + 1, c2 = threadIdx.y + 1, c3 = threadIdx.z + 1;
  int globalIdx = dev_mesh.guard[0] + t1 * DIM1 + c1 - 1 +
                  (dev_mesh.guard[1] + t2 * DIM2 + c2 - 1) * dev_mesh.dims[0] +
                  (dev_mesh.guard[2] + t3 * DIM3 + c3 - 1) *
                  dev_mesh.dims[0] * dev_mesh.dims[1];
  init_shared_memory<DIM1, DIM2, DIM3>(s_u1, s_u2, s_u3, u1, u2, u3,
                                       globalIdx, c1, c2, c3);
  __syncthreads();

  // Do the actual computation here
  // (Curl u)_1 = d2u3 - d3u2
  v1[globalIdx] += d2<DIM1, DIM2, DIM3>(s_u3, c1, c2 + flip(s3[1]), c3) -
                   d3<DIM1, DIM2, DIM3>(s_u2, c1, c2, c3 + flip(s2[2]));
  // v1[globalIdx] = (s_u3[c3][c2 + flip(s3[1])][c1] - s_u3[c3][c2 - 1 + flip(s3[1])][c1]) / dev_mesh.delta[1] -
  //                 (s_u2[c3 + flip(s2[2])][c2][c1] - s_u2[c3 - 1 + flip(s2[2])][c2][c1]) / dev_mesh.delta[2];
  // (Curl u)_2 = d3u1 - d1u3
  v2[globalIdx] += d3<DIM1, DIM2, DIM3>(s_u1, c1, c2, c3 + flip(s1[2])) -
                   d1<DIM1, DIM2, DIM3>(s_u3, c1 + flip(s3[0]), c2, c3);
  // v2[globalIdx] = (s_u1[c3 + flip(s1[2])][c2][c1] - s_u1[c3 - 1 + flip(s1[2])][c2][c1]) / dev_mesh.delta[2] -
  //                 (s_u3[c3][c2][c1 + flip(s3[0])] - s_u3[c3][c2][c1 - 1 + flip(s3[0])]) / dev_mesh.delta[0];

  // (Curl u)_3 = d1u2 - d2u1
  v3[globalIdx] += d1<DIM1, DIM2, DIM3>(s_u2, c1 + flip(s2[0]), c2, c3) -
                   d2<DIM1, DIM2, DIM3>(s_u1, c1, c2 + flip(s1[1]), c3);
  // v3[globalIdx] = (s_u2[c3][c2][c1 + flip(s2[0])] - s_u2[c3][c2][c1 - 1 + flip(s2[0])]) / dev_mesh.delta[0] -
  //                 (s_u1[c3][c2 + flip(s1[1])][c1] - s_u1[c3][c2 - 1 + flip(s1[1])][c1]) / dev_mesh.delta[1];
}

template <int DIM1, int DIM2, int DIM3>
__global__
void compute_div(Scalar* v, const Scalar* u1, const Scalar* u2, const Scalar* u3,
                 Stagger s1, Stagger s2, Stagger s3) {
  // Declare cache array in shared memory
  __shared__ Scalar s_u1[DIM3 + 2][DIM2 + 2][DIM1 + 2];
  __shared__ Scalar s_u2[DIM3 + 2][DIM2 + 2][DIM1 + 2];
  __shared__ Scalar s_u3[DIM3 + 2][DIM2 + 2][DIM1 + 2];

  // Load indices
  int t1 = blockIdx.x, t2 = blockIdx.y, t3 = blockIdx.z;
  int c1 = threadIdx.x + 1, c2 = threadIdx.y + 1, c3 = threadIdx.z + 1;
  int globalIdx = dev_mesh.guard[0] + t1 * DIM1 + c1-1 +
                  (dev_mesh.guard[1] + t2 * DIM2 + c2-1) * dev_mesh.dims[0] +
                  (dev_mesh.guard[2] + t3 * DIM3 + c3-1) *
                  dev_mesh.dims[0] * dev_mesh.dims[1];
  init_shared_memory<DIM1, DIM2, DIM3>(s_u1, s_u2, s_u3, u1, u2, u3,
                                       globalIdx, c1, c2, c3);
  __syncthreads();

  // Do the actual computation here
  v[globalIdx] += d1<DIM1, DIM2, DIM3>(s_u1, c1 + flip(s1[0]), c2, c3) +
                  d2<DIM1, DIM2, DIM3>(s_u2, c1, c2 + flip(s2[1]), c3) +
                  d3<DIM1, DIM2, DIM3>(s_u3, c1, c2, c3 + flip(s3[2]));
}

template <int DIM1, int DIM2, int DIM3>
__global__
void compute_grad(Scalar* v1, Scalar* v2, Scalar* v3,
                  const Scalar* u, Stagger s) {
  // Declare cache array in shared memory
  __shared__ Scalar s_u[DIM3 + 2][DIM2 + 2][DIM1 + 2];

  // Load indices
  int t1 = blockIdx.x, t2 = blockIdx.y, t3 = blockIdx.z;
  int c1 = threadIdx.x + 1, c2 = threadIdx.y + 1, c3 = threadIdx.z + 1;
  int globalIdx = dev_mesh.guard[0] + t1 * DIM1 + c1-1 +
                  (dev_mesh.guard[1] + t2 * DIM2 + c2-1) * dev_mesh.dims[0] +
                  (dev_mesh.guard[2] + t3 * DIM3 + c3-1) *
                  dev_mesh.dims[0] * dev_mesh.dims[1];

  // Load field values into shared memory
  s_u[c3][c2][c1] = u[globalIdx];

  // Handle extra guard cells
  if (c1 == 1) {
    s_u[c3][c2][c1 - 1] = u[globalIdx - 1];
  } else if (c1 == DIM1) {
    s_u[c3][c2][c1 + 1] = u[globalIdx + 1];
  }
  if (c2 == 1) {
    s_u[c3][c2 - 1][c1] = u[globalIdx - dev_mesh.dims[0]];
  } else if (c2 == DIM2) {
    s_u[c3][c2 + 1][c1] = u[globalIdx + dev_mesh.dims[0]];
  }
  if (c3 == 1) {
    s_u[c3 - 1][c2][c1] = u[globalIdx - dev_mesh.dims[0] * dev_mesh.dims[1]];
  } else if (c3 == DIM3) {
    s_u[c3 + 1][c2][c1] = u[globalIdx + dev_mesh.dims[0] * dev_mesh.dims[1]];
  }
  __syncthreads();

  v1[globalIdx] += d1<DIM1, DIM2, DIM3>(s_u, c1 + flip(s[0]), c2, c3);
  v2[globalIdx] += d2<DIM1, DIM2, DIM3>(s_u, c1, c2 + flip(s[1]), c3);
  v3[globalIdx] += d3<DIM1, DIM2, DIM3>(s_u, c1, c2, c3 + flip(s[2]));
}

}

void curl(VectorField<Scalar>& result, const VectorField<Scalar>& u) {
  auto& grid = u.grid();
  auto& mesh = grid.mesh();

  // TODO: reset the result first?

  // TODO: The kernel launch parameters might need some tuning for different
  // architectures

  dim3 blockSize(8, 8, 8);
  dim3 gridSize(mesh.reduced_dim(0) / 8, mesh.reduced_dim(1) / 8,
                mesh.reduced_dim(2) / 8);
  Kernels::compute_curl<8, 8, 8><<<gridSize, blockSize>>>
      (result.ptr(0), result.ptr(1), result.ptr(2),
       u.ptr(0), u.ptr(1), u.ptr(2),
       u.stagger(0), u.stagger(1), u.stagger(2));
  CudaCheckError();
}

void div(ScalarField<Scalar>& result, const VectorField<Scalar>& u) {
  auto& grid = u.grid();
  auto& mesh = grid.mesh();

  // TODO: reset the result first?

  // TODO: The kernel launch parameters might need some tuning for different
  // architectures

  dim3 blockSize(8, 8, 8);
  dim3 gridSize(mesh.reduced_dim(0) / 8, mesh.reduced_dim(1) / 8,
                mesh.reduced_dim(2) / 8);
  Kernels::compute_div<8, 8, 8><<<gridSize, blockSize>>>
      (result.ptr(), u.ptr(0), u.ptr(1), u.ptr(2),
       u.stagger(0), u.stagger(1), u.stagger(2));
  CudaCheckError();
}

void grad(VectorField<Scalar>& result, const ScalarField<Scalar>& u) {
  auto& grid = u.grid();
  auto& mesh = grid.mesh();

  // TODO: reset the result first?

  // TODO: The kernel launch parameters might need some tuning for different
  // architectures

  dim3 blockSize(8, 8, 8);
  dim3 gridSize(mesh.reduced_dim(0) / 8, mesh.reduced_dim(1) / 8,
                mesh.reduced_dim(2) / 8);
  Kernels::compute_grad<8, 8, 8><<<gridSize, blockSize>>>
      (result.ptr(0), result.ptr(1), result.ptr(2), u.ptr(),
       u.stagger());
  CudaCheckError();
}

}
