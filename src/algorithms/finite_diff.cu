#include "algorithms/finite_diff.h"
#include "data/stagger.h"
#include "cuda/constant_mem.h"
#include "cuda/cudaUtility.h"
#include "utils/util_functions.h"

namespace Aperture {

namespace Kernels {

template <int DIM1, int DIM2, int DIM3>
__global__
void curl_impl(Scalar* v1, Scalar* v2, Scalar* v3,
               const Scalar* u1, const Scalar* u2, const Scalar* u3,
               Stagger s1, Stagger s2, Stagger s3) {
  // Declare cache array in shared memory
  __shared__ Scalar s_u1[DIM3 + 2][DIM2 + 2][DIM1 + 2];
  __shared__ Scalar s_u2[DIM3 + 2][DIM2 + 2][DIM1 + 2];
  __shared__ Scalar s_u3[DIM3 + 2][DIM2 + 2][DIM1 + 2];

  // Load indices
  int t1 = blockIdx.x, t2 = blockIdx.y, t3 = blockIdx.z;
  int c1 = threadIdx.x, c2 = threadIdx.y, c3 = threadIdx.z;
  int globalIdx = dev_mesh.guard[0] + t1 * DIM1 + c1 +
                  (dev_mesh.guard[1] + t2 * DIM2 + c2) * dev_mesh.dims[0] +
                  (dev_mesh.guard[2] + t3 * DIM3 + c3) *
                  dev_mesh.dims[0] * dev_mesh.dims[1];

  // Load field values into shared memory
  s_u1[c3 + 1][c2 + 1][c1 + 1] = u1[globalIdx];
  s_u2[c3 + 1][c2 + 1][c1 + 1] = u2[globalIdx];
  s_u3[c3 + 1][c2 + 1][c1 + 1] = u3[globalIdx];

  // Handle extra guard cells
  if (c1 == 0) {
    s_u1[c3 + 1][c2 + 1][c1] = u1[globalIdx - 1];
    s_u2[c3 + 1][c2 + 1][c1] = u2[globalIdx - 1];
    s_u3[c3 + 1][c2 + 1][c1] = u3[globalIdx - 1];
  } else if (c1 == DIM1 - 1) {
    s_u1[c3 + 1][c2 + 1][c1 + 2] = u1[globalIdx + 1];
    s_u2[c3 + 1][c2 + 1][c1 + 2] = u2[globalIdx + 1];
    s_u3[c3 + 1][c2 + 1][c1 + 2] = u3[globalIdx + 1];
  }
  if (c2 == 0) {
    s_u1[c3 + 1][c2][c1 + 1] = u1[globalIdx - dev_mesh.dims[0]];
    s_u2[c3 + 1][c2][c1 + 1] = u2[globalIdx - dev_mesh.dims[0]];
    s_u3[c3 + 1][c2][c1 + 1] = u3[globalIdx - dev_mesh.dims[0]];
  } else if (c2 == DIM2 - 1) {
    s_u1[c3 + 1][c2 + 2][c1 + 1] = u1[globalIdx + dev_mesh.dims[0]];
    s_u2[c3 + 1][c2 + 2][c1 + 1] = u2[globalIdx + dev_mesh.dims[0]];
    s_u3[c3 + 1][c2 + 2][c1 + 1] = u3[globalIdx + dev_mesh.dims[0]];
  }
  if (c3 == 0) {
    s_u1[c3][c2 + 1][c1 + 1] = u1[globalIdx - dev_mesh.dims[0] * dev_mesh.dims[1]];
    s_u2[c3][c2 + 1][c1 + 1] = u2[globalIdx - dev_mesh.dims[0] * dev_mesh.dims[1]];
    s_u3[c3][c2 + 1][c1 + 1] = u3[globalIdx - dev_mesh.dims[0] * dev_mesh.dims[1]];
  } else if (c3 == DIM3 - 1) {
    s_u1[c3 + 2][c2 + 1][c1 + 1] = u1[globalIdx + dev_mesh.dims[0] * dev_mesh.dims[1]];
    s_u2[c3 + 2][c2 + 1][c1 + 1] = u2[globalIdx + dev_mesh.dims[0] * dev_mesh.dims[1]];
    s_u3[c3 + 2][c2 + 1][c1 + 1] = u3[globalIdx + dev_mesh.dims[0] * dev_mesh.dims[1]];
  }
  __syncthreads();

  // Do the actual computation here?
  // (Curl u)_1 = d2u3 - d3u2
  v1[globalIdx] = (s_u3[c3][c2 + flip(s3[1])][c1] - s_u3[c3][c2 - 1 + flip(s3[1])][c1]) / dev_mesh.delta[1] -
                  (s_u2[c3 + flip(s2[2])][c2][c1] - s_u2[c3 - 1 + flip(s2[2])][c2][c1]) / dev_mesh.delta[2];
  // (Curl u)_2 = d3u1 - d1u3
  v2[globalIdx] = (s_u1[c3 + flip(s1[2])][c2][c1] - s_u1[c3 - 1 + flip(s1[2])][c2][c1]) / dev_mesh.delta[2] -
                  (s_u3[c3][c2][c1 + flip(s3[0])] - s_u3[c3][c2][c1 - 1 + flip(s3[0])]) / dev_mesh.delta[0];

  // (Curl u)_3 = d1u2 - d2u1
  v3[globalIdx] = (s_u2[c3][c2][c1 + flip(s2[0])] - s_u2[c3][c2][c1 - 1 + flip(s2[0])]) / dev_mesh.delta[0] -
                  (s_u1[c3][c2 + flip(s1[1])][c1] - s_u1[c3][c2 - 1 + flip(s1[1])][c1]) / dev_mesh.delta[1];
}

}

void curl(VectorField<Scalar>& result, const VectorField<Scalar>& u) {
  auto& grid = u.grid();
  auto& mesh = grid.mesh();
  dim3 blockSize(8, 8, 8);
  dim3 gridSize(mesh.reduced_dim(0) / 8, mesh.reduced_dim(1) / 8,
                mesh.reduced_dim(2) / 8);
  Kernels::curl_impl<8, 8, 8><<<gridSize, blockSize>>>
      (result.ptr(0), result.ptr(1), result.ptr(2),
       u.ptr(0), u.ptr(1), u.ptr(2),
       u.stagger(0), u.stagger(1), u.stagger(2));
  CudaCheckError();
}

}
