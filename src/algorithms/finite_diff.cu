#include "algorithms/finite_diff.h"
#include "data/stagger.h"
#include "cuda/constant_mem.h"
#include "cuda/cudaUtility.h"
#include "utils/util_functions.h"

namespace Aperture {

namespace Kernels {

#define PAD 1

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

template <int TILE1, int TILE2>
__global__
void deriv_x(cudaPitchedPtr df, cudaPitchedPtr f, int stagger, Scalar q) {
  __shared__ Scalar s_f[TILE2][TILE1 + PAD*2];

  int i = threadIdx.x + blockIdx.x * blockDim.x + dev_mesh.guard[0];
      // j = threadIdx.y;
  int si = threadIdx.x + PAD;
  int t2 = blockIdx.y * blockDim.y + dev_mesh.guard[1],
       k = blockIdx.z + dev_mesh.guard[2],
      t3 = (k * f.ysize + (threadIdx.y + t2)) * f.pitch;

  Scalar* row = (Scalar*)((char*)f.ptr + t3);
  Scalar* row_df = (Scalar*)((char*)df.ptr + t3);
  // int offset = 1 + i;

  s_f[threadIdx.y][si] = row[i];
  // __syncthreads();

  // Fill the boundary guard cells
  if (si < PAD * 2) {
    s_f[threadIdx.y][si - PAD] = row[i - PAD];
  // } else if (i >= TILE1 - PAD) {
    s_f[threadIdx.y][si + TILE1] = row[i + TILE1];
  }
  __syncthreads();

  // compute the derivative
  row_df[i] += (s_f[threadIdx.y][si + stagger] -
                s_f[threadIdx.y][si + stagger - 1]) * q /
               dev_mesh.delta[0];
}

template <int TILE1, int TILE2>
__global__
void deriv_y(cudaPitchedPtr df, cudaPitchedPtr f, Stagger s, Scalar q) {
  __shared__ Scalar s_f[TILE2 + PAD*2][TILE1];

  int i = threadIdx.x;
  int t1 = blockIdx.x * blockDim.x,
      t2 = TILE2 * blockIdx.y,
       k = blockIdx.z;

  for (int j = threadIdx.y; j < TILE2; j += blockDim.y) {
    int sj = j + PAD;
    size_t globalOffset = (dev_mesh.guard[2] + k) * f.pitch * f.ysize +
                          (dev_mesh.guard[1] + j + t2) * f.pitch +
                          (dev_mesh.guard[0] + i + t1) * sizeof(Scalar);
    s_f[sj][i] = *(Scalar*)((char*)f.ptr + globalOffset);
  }

  __syncthreads();

  // Fill the guard cells
  if (threadIdx.y < PAD) {
    size_t offset = (dev_mesh.guard[2] + k) * f.pitch * f.ysize +
                    (dev_mesh.guard[1] + threadIdx.y - PAD + t2) * f.pitch +
                    (dev_mesh.guard[0] + i + t1) * sizeof(Scalar);
    s_f[threadIdx.y][i] = *(Scalar*)((char*)f.ptr + offset);
  // } else if (threadIdx.y >= blockDim.y - PAD) {
    offset = (dev_mesh.guard[2] + k) * f.pitch * f.ysize +
             (dev_mesh.guard[1] + TILE2 + threadIdx.y + t2) * f.pitch +
             (dev_mesh.guard[0] + i + t1) * sizeof(Scalar);
    s_f[TILE2 + PAD + threadIdx.y][i] = *(Scalar*)((char*)f.ptr + offset);
  }

  __syncthreads();

  // compute the derivative
  for (int j = threadIdx.y; j < TILE2; j += blockDim.y) {
    int sj = j + PAD;
    size_t globalOffset = (dev_mesh.guard[2] + k) * f.pitch * f.ysize +
                          (dev_mesh.guard[1] + j + t2) * f.pitch +
                          (dev_mesh.guard[0] + i + t1) * sizeof(Scalar);
    (*(Scalar*)((char*)df.ptr + globalOffset)) += deriv(s_f[sj + flip(s[1]) - 1][i],
                                                        s_f[sj + flip(s[1])][i],
                                                        dev_mesh.delta[1]) * q;
  }
}

template <int DIM1, int DIM2, int DIM3>
__device__ __forceinline__
void init_shared_memory(Scalar s_u1[][DIM2 + 2][DIM1 + 2],
                        Scalar s_u2[][DIM2 + 2][DIM1 + 2],
                        Scalar s_u3[][DIM2 + 2][DIM1 + 2],
                        cudaPitchedPtr& u1, cudaPitchedPtr& u2, cudaPitchedPtr& u3,
                        size_t globalOffset, int c1, int c2, int c3) {
  // Load field values into shared memory
  s_u1[c3][c2][c1] = *(Scalar*)((char*)u1.ptr + globalOffset);
  s_u2[c3][c2][c1] = *(Scalar*)((char*)u2.ptr + globalOffset);
  s_u3[c3][c2][c1] = *(Scalar*)((char*)u3.ptr + globalOffset);

  // Handle extra guard cells
  if (c1 == 1) {
    s_u1[c3][c2][c1 - 1] = *(Scalar*)((char*)u1.ptr + globalOffset - sizeof(Scalar));
    s_u2[c3][c2][c1 - 1] = *(Scalar*)((char*)u2.ptr + globalOffset - sizeof(Scalar));
    s_u3[c3][c2][c1 - 1] = *(Scalar*)((char*)u3.ptr + globalOffset - sizeof(Scalar));
  } else if (c1 == DIM1) {
    s_u1[c3][c2][c1 + 1] = *(Scalar*)((char*)u1.ptr + globalOffset + sizeof(Scalar));
    s_u2[c3][c2][c1 + 1] = *(Scalar*)((char*)u2.ptr + globalOffset + sizeof(Scalar));
    s_u3[c3][c2][c1 + 1] = *(Scalar*)((char*)u3.ptr + globalOffset + sizeof(Scalar));
  }
  if (c2 == 1) {
    s_u1[c3][c2 - 1][c1] = *(Scalar*)((char*)u1.ptr + globalOffset - u1.pitch);
    s_u2[c3][c2 - 1][c1] = *(Scalar*)((char*)u2.ptr + globalOffset - u2.pitch);
    s_u3[c3][c2 - 1][c1] = *(Scalar*)((char*)u3.ptr + globalOffset - u3.pitch);
  } else if (c2 == DIM2) {
    s_u1[c3][c2 + 1][c1] = *(Scalar*)((char*)u1.ptr + globalOffset + u1.pitch);
    s_u2[c3][c2 + 1][c1] = *(Scalar*)((char*)u2.ptr + globalOffset + u2.pitch);
    s_u3[c3][c2 + 1][c1] = *(Scalar*)((char*)u3.ptr + globalOffset + u3.pitch);
  }
  if (c3 == 1) {
    s_u1[c3 - 1][c2][c1] = *(Scalar*)((char*)u1.ptr + globalOffset - u1.pitch * u1.ysize);
    s_u2[c3 - 1][c2][c1] = *(Scalar*)((char*)u2.ptr + globalOffset - u2.pitch * u2.ysize);
    s_u3[c3 - 1][c2][c1] = *(Scalar*)((char*)u3.ptr + globalOffset - u3.pitch * u3.ysize);
  } else if (c3 == DIM3) {
    s_u1[c3 + 1][c2][c1] = *(Scalar*)((char*)u1.ptr + globalOffset + u1.pitch * u1.ysize);
    s_u2[c3 + 1][c2][c1] = *(Scalar*)((char*)u2.ptr + globalOffset + u2.pitch * u2.ysize);
    s_u3[c3 + 1][c2][c1] = *(Scalar*)((char*)u3.ptr + globalOffset + u3.pitch * u3.ysize);
  }
}

template <int DIM1, int DIM2, int DIM3>
__global__
void compute_curl(cudaPitchedPtr v1, cudaPitchedPtr v2, cudaPitchedPtr v3,
                  cudaPitchedPtr u1, cudaPitchedPtr u2, cudaPitchedPtr u3,
                  Stagger s1, Stagger s2, Stagger s3) {
  // Declare cache array in shared memory
  __shared__ Scalar s_u1[DIM3 + 2][DIM2 + 2][DIM1 + 2];
  __shared__ Scalar s_u2[DIM3 + 2][DIM2 + 2][DIM1 + 2];
  __shared__ Scalar s_u3[DIM3 + 2][DIM2 + 2][DIM1 + 2];

  // Load shared memory
  int c1 = threadIdx.x + 1, c2 = threadIdx.y + 1, c3 = threadIdx.z + 1;
  int t1 = blockIdx.x, t2 = blockIdx.y, t3 = blockIdx.z;
  size_t globalOffset =  (dev_mesh.guard[2] + t3 * DIM3 + c3 - 1) * u1.pitch * u1.ysize +
                         (dev_mesh.guard[1] + t2 * DIM2 + c2 - 1) * u1.pitch +
                         (dev_mesh.guard[0] + t1 * DIM1 + c1 - 1) * sizeof(Scalar);

  init_shared_memory<DIM1, DIM2, DIM3>(s_u1, s_u2, s_u3, u1, u2, u3,
                                       globalOffset, c1, c2, c3);
  // for (int offset3 = 0; offset3 < DIM3; offset3 += blockDim.z) {
  //   for (int offset2 = 0; offset2 < DIM2; offset2 += blockDim.y) {
  //     for (int offset1 = 0; offset1 < DIM1; offset1 += blockDim.x) {
  //       globalIdx = dev_mesh.guard[0] + t1 * DIM1 + c1 - 1 + offset1 +
  //                   (dev_mesh.guard[1] + t2 * DIM2 + c2 - 1 + offset2) *
  //                   dev_mesh.dims[0] +
  //                   (dev_mesh.guard[2] + t3 * DIM3 + c3 - 1 + offset3) *
  //                   dev_mesh.dims[0] * dev_mesh.dims[1];
  //       init_shared_memory<DIM1, DIM2, DIM3>(s_u1, s_u2, s_u3, u1, u2, u3,
  //                                            globalIdx, c1 + offset1,
  //                                            c2 + offset2,
  //                                            c3 + offset3);
  //     }
  //   }
  // }
  __syncthreads();

  // Do the actual computation here
  // for (int offset3 = 0; offset3 < DIM3; offset3 += blockDim.z) {
  //   for (int offset2 = 0; offset2 < DIM2; offset2 += blockDim.y) {
  //     for (int offset1 = 0; offset1 < DIM1; offset1 += blockDim.x) {
  //       globalIdx = dev_mesh.guard[0] + t1 * DIM1 + c1 - 1 + offset1 +
  //                   (dev_mesh.guard[1] + t2 * DIM2 + c2 - 1 + offset2) *
  //                   dev_mesh.dims[0] +
  //                   (dev_mesh.guard[2] + t3 * DIM3 + c3 - 1 + offset3) *
  //                   dev_mesh.dims[0] * dev_mesh.dims[1];
  // (Curl u)_1 = d2u3 - d3u2
  (*(Scalar*)((char*)v1.ptr + globalOffset)) +=
      d2<DIM1, DIM2, DIM3>(s_u3, c1, c2 + flip(s3[1]), c3) -
      d3<DIM1, DIM2, DIM3>(s_u2, c1, c2, c3 + flip(s2[2]));
  // v1[globalIdx] = (s_u3[c3][c2 + flip(s3[1])][c1] - s_u3[c3][c2 - 1 + flip(s3[1])][c1]) / dev_mesh.delta[1] -
  //                 (s_u2[c3 + flip(s2[2])][c2][c1] - s_u2[c3 - 1 + flip(s2[2])][c2][c1]) / dev_mesh.delta[2];
  // (Curl u)_2 = d3u1 - d1u3
  (*(Scalar*)((char*)v1.ptr + globalOffset)) +=
      d3<DIM1, DIM2, DIM3>(s_u1, c1, c2, c3 + flip(s1[2])) -
      d1<DIM1, DIM2, DIM3>(s_u3, c1 + flip(s3[0]), c2, c3);
  // v2[globalIdx] = (s_u1[c3 + flip(s1[2])][c2][c1] - s_u1[c3 - 1 + flip(s1[2])][c2][c1]) / dev_mesh.delta[2] -
  //                 (s_u3[c3][c2][c1 + flip(s3[0])] - s_u3[c3][c2][c1 - 1 + flip(s3[0])]) / dev_mesh.delta[0];

  // (Curl u)_3 = d1u2 - d2u1
  (*(Scalar*)((char*)v1.ptr + globalOffset)) +=
      d1<DIM1, DIM2, DIM3>(s_u2, c1 + flip(s2[0]), c2, c3) -
      d2<DIM1, DIM2, DIM3>(s_u1, c1, c2 + flip(s1[1]), c3);
  // v3[globalIdx] = (s_u2[c3][c2][c1 + flip(s2[0])] - s_u2[c3][c2][c1 - 1 + flip(s2[0])]) / dev_mesh.delta[0] -
  //                 (s_u1[c3][c2 + flip(s1[1])][c1] - s_u1[c3][c2 - 1 + flip(s1[1])][c1]) / dev_mesh.delta[1];
  //     }
  //   }
  // }
}

template <int DIM1, int DIM2, int DIM3>
__global__
void compute_div(cudaPitchedPtr v, cudaPitchedPtr u1,
                 cudaPitchedPtr u2, cudaPitchedPtr u3,
                 Stagger s1, Stagger s2, Stagger s3) {
  // Declare cache array in shared memory
  __shared__ Scalar s_u1[DIM3 + 2][DIM2 + 2][DIM1 + 2];
  __shared__ Scalar s_u2[DIM3 + 2][DIM2 + 2][DIM1 + 2];
  __shared__ Scalar s_u3[DIM3 + 2][DIM2 + 2][DIM1 + 2];

  // Load indices
  int t1 = blockIdx.x, t2 = blockIdx.y, t3 = blockIdx.z;
  int c1 = threadIdx.x + 1, c2 = threadIdx.y + 1, c3 = threadIdx.z + 1;
  size_t globalOffset =  + (dev_mesh.guard[2] + t3 * DIM3 + c3 - 1) * u1.pitch * u1.ysize
                         + (dev_mesh.guard[1] + t2 * DIM2 + c2 - 1) * u1.pitch
                         + (dev_mesh.guard[0] + t1 * DIM1 + c1 - 1) * sizeof(Scalar);

  init_shared_memory<DIM1, DIM2, DIM3>(s_u1, s_u2, s_u3, u1, u2, u3,
                                       globalOffset, c1, c2, c3);
  __syncthreads();

  // Do the actual computation here
  (*(Scalar*)((char*)v.ptr + globalOffset)) += d1<DIM1, DIM2, DIM3>(s_u1, c1 + flip(s1[0]), c2, c3) +
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

  dim3 blockSize(64, 8, 1);
  dim3 gridSize(mesh.reduced_dim(0) / 64, mesh.reduced_dim(1) / 8,
                mesh.reduced_dim(2));
  // Kernels::compute_curl<16, 8, 8><<<gridSize, blockSize>>>
  //     (result.ptr(0), result.ptr(1), result.ptr(2),
  //      u.ptr(0), u.ptr(1), u.ptr(2),
  //      u.stagger(0), u.stagger(1), u.stagger(2));
  Kernels::deriv_x<64, 8><<<gridSize, blockSize>>>
      (result.ptr(2), u.ptr(1), flip(u.stagger(1)[0]), 1.0);
  CudaCheckError();

  // blockSize = dim3(32, 16, 1);
  // gridSize = dim3(mesh.reduced_dim(0) / 32, mesh.reduced_dim(1) / 64,
  //                 mesh.reduced_dim(2));
  // Kernels::deriv_y<32, 64><<<gridSize, blockSize>>>
  //     (result.ptr(2), u.ptr(0), u.stagger(0), -1.0);
  // CudaCheckError();
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
      (result.data().data_d(), u.ptr(0), u.ptr(1), u.ptr(2),
       u.stagger(0), u.stagger(1), u.stagger(2));
  CudaCheckError();
}

// void grad(VectorField<Scalar>& result, const ScalarField<Scalar>& u) {
//   auto& grid = u.grid();
//   auto& mesh = grid.mesh();

//   // TODO: reset the result first?

//   // TODO: The kernel launch parameters might need some tuning for different
//   // architectures

//   dim3 blockSize(8, 8, 8);
//   dim3 gridSize(mesh.reduced_dim(0) / 8, mesh.reduced_dim(1) / 8,
//                 mesh.reduced_dim(2) / 8);
//   Kernels::compute_grad<8, 8, 8><<<gridSize, blockSize>>>
//       (result.ptr(0), result.ptr(1), result.ptr(2), u.ptr(),
//        u.stagger());
//   CudaCheckError();
// }

}
