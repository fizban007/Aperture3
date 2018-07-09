#include "algorithms/finite_diff.h"
#include "data/stagger.h"
#include "cuda/constant_mem.h"
#include "cuda/cudaUtility.h"
#include "utils/util_functions.h"

namespace Aperture {

// It is not worth using 4th order differentiation when using float accuracy
constexpr int order = 2;

namespace Kernels {

// #define PAD 1
template <int Order> struct Pad;

template <>
struct Pad<2> { enum { val = 1 }; };

template <>
struct Pad<4> { enum { val = 2 }; };

template <>
struct Pad<6> { enum { val = 3 }; };

template <int Order>
HD_INLINE
Scalar deriv(Scalar f[], Scalar delta);
  
template <>
HD_INLINE
Scalar deriv<2>(Scalar f[], Scalar delta) {
  return (f[1] - f[0]) / delta;
}

template <>
HD_INLINE
Scalar deriv<4>(Scalar f[], Scalar delta) {
  return ((f[2] - f[1]) * 1.125f - (f[3] - f[0]) * 0.041666667f) / delta;
}

template <>
HD_INLINE
Scalar deriv<6>(Scalar f[], Scalar delta) {
  // TODO: Add support for 6th order differentiation
  return ((f[2] - f[1]) * 1.125f - (f[3] - f[0]) * 0.041666667f) / delta;
}

template <int Order, int DIM1, int DIM2, int DIM3>
__device__ __forceinline__
Scalar d1(Scalar array[][DIM2 + Pad<Order>::val*2][DIM1 + Pad<Order>::val*2],
          int c1, int c2, int c3) {
  Scalar val[Order];
#pragma unroll
  for (int i = 0; i < Order; i++)
    val[i] = array[c3][c2][c1 - Pad<Order>::val + i];
  return deriv<Order>(val, dev_mesh.delta[0]);
}

template <int Order, int DIM1, int DIM2, int DIM3>
__device__ __forceinline__
Scalar d2(Scalar array[][DIM2 + Pad<Order>::val*2][DIM1 + Pad<Order>::val*2],
          int c1, int c2, int c3) {
  Scalar val[Order];
#pragma unroll
  for (int i = 0; i < Order; i++)
    val[i] = array[c3][c2 - Pad<Order>::val + i][c1];
  return deriv<Order>(val, dev_mesh.delta[1]);
}

template <int Order, int DIM1, int DIM2, int DIM3>
__device__ __forceinline__
Scalar d3(Scalar array[][DIM2 + Pad<Order>::val*2][DIM1 + Pad<Order>::val*2],
          int c1, int c2, int c3) {
  Scalar val[Order];
#pragma unroll
  for (int i = 0; i < Order; i++)
    val[i] = array[c3 - Pad<Order>::val + i][c2][c1];
  return deriv<Order>(val, dev_mesh.delta[2]);
}

template <int Order, int DIM1>
__device__ __forceinline__
Scalar dx(Scalar array[][DIM1], int c1, int c2, Scalar delta) {
  Scalar val[Order];
#pragma unroll
  for (int i = 0; i < Order; i++)
    val[i] = array[c2][c1 - Pad<Order>::val + i];
  return deriv<Order>(val, delta);
}

template <int Order, int DIM1>
__device__ __forceinline__
Scalar dy(Scalar array[][DIM1], int c1, int c2, Scalar delta) {
  Scalar val[Order];
#pragma unroll
  for (int i = 0; i < Order; i++)
    val[i] = array[c2 - Pad<Order>::val + i][c1];
  return deriv<Order>(val, delta);
}

template <int Order, int DIM1, int DIM2>
__global__
void deriv_x(cudaPitchedPtr df, cudaPitchedPtr f, int stagger, Scalar q) {
  __shared__ Scalar s_f[DIM2][DIM1 + Pad<Order>::val * 2];

  // Load indices
  int i = threadIdx.x + blockIdx.x * blockDim.x + dev_mesh.guard[0];
  int si = threadIdx.x + Pad<Order>::val;
  size_t globalOffset = ((blockIdx.z + dev_mesh.guard[2]) * f.ysize +
                      (threadIdx.y + blockIdx.y * blockDim.y +
                       dev_mesh.guard[1])) * f.pitch;

  // Read data into shared memory
  Scalar* row = (Scalar*)((char*)f.ptr + globalOffset);
  Scalar* row_df = (Scalar*)((char*)df.ptr + globalOffset);

  s_f[threadIdx.y][si] = row[i];

  // Fill the boundary guard cells
  if (si < Pad<Order>::val * 2) {
    s_f[threadIdx.y][si - Pad<Order>::val] = row[i - Pad<Order>::val];
    s_f[threadIdx.y][si + DIM1] = row[i + DIM1];
  }
  __syncthreads();

  // compute the derivative
  // row_df[i] += (s_f[threadIdx.y][si + stagger] -
  //               s_f[threadIdx.y][si + stagger - 1]) * q /
  //              dev_mesh.delta[0];
  row_df[i] += q * dx<Order>(s_f, si + stagger, threadIdx.y, dev_mesh.delta[0]);
}

template <int Order, int DIM1, int DIM2>
__global__
void deriv_y(cudaPitchedPtr df, cudaPitchedPtr f, int stagger, Scalar q) {
  __shared__ Scalar s_f[DIM2 + Pad<Order>::val*2][DIM1];

  // Load indices
  int i = threadIdx.x + blockIdx.x * blockDim.x + dev_mesh.guard[0];
  long offset = ((DIM2 * blockIdx.y) +
                   (blockIdx.z + dev_mesh.guard[2]) * f.ysize) * f.pitch;

  // Read data into shared memory
  for (int j = threadIdx.y; j < DIM2; j += blockDim.y) {
    int sj = j + Pad<Order>::val;
    Scalar* row = (Scalar*)((char*)f.ptr + offset
                            + (j + dev_mesh.guard[1]) * f.pitch);
    s_f[sj][threadIdx.x] = row[i];
  }

  // Fill the guard cells
  if (threadIdx.y < Pad<Order>::val) {
    // if (threadIdx.y == 0 && blockIdx.y == 0)
    //   printf("Diff is %d\n", offset + (threadIdx.y - Pad<Order>::val) * (int)f.pitch);
    Scalar* row = (Scalar*)((char*)f.ptr + offset + (threadIdx.y - Pad<Order>::val + dev_mesh.guard[1]) * f.pitch);
    s_f[threadIdx.y][threadIdx.x] = row[i];
    // s_f[threadIdx.y][threadIdx.x] = 0.0f;
    row = (Scalar*)((char*)f.ptr + offset + (threadIdx.y + DIM2 + dev_mesh.guard[1]) * f.pitch);
    s_f[DIM2 + Pad<Order>::val + threadIdx.y][threadIdx.x] = row[i];
    // s_f[DIM2 + Pad<Order>::val + threadIdx.y][threadIdx.x] = 0.0f;
  }

  __syncthreads();

  // compute the derivative
  for (int j = threadIdx.y; j < DIM2; j += blockDim.y) {
    int sj = j + Pad<Order>::val;
    Scalar* row_df = (Scalar*)((char*)df.ptr + (j + dev_mesh.guard[1]) * df.pitch + offset);
    // row_df[i] += (s_f[sj + stagger][threadIdx.x] -
    //               s_f[sj + stagger - 1][threadIdx.x]) * q /
    //              dev_mesh.delta[1];
    row_df[i] += q * dy<Order>(s_f, threadIdx.x, sj + stagger, dev_mesh.delta[1]);
  }
}

template <int Order, int DIM1, int DIM3>
__global__
void deriv_z(cudaPitchedPtr df, cudaPitchedPtr f, int stagger, Scalar q) {
  __shared__ Scalar s_f[DIM3 + Pad<Order>::val*2][DIM1];

  // Load indices
  int i = threadIdx.x + blockIdx.x * blockDim.x + dev_mesh.guard[0];
  int dz = f.ysize * f.pitch;
  long offset = (blockIdx.z + dev_mesh.guard[1]) * f.pitch +
               (DIM3 * blockIdx.y) * dz;
               
  // Read data into shared memory
  for (int j = threadIdx.y; j < DIM3; j += blockDim.y) {
    int sj = j + Pad<Order>::val;
    Scalar* row = (Scalar*)((char*)f.ptr + offset
                            + (j + dev_mesh.guard[2]) * dz);
    s_f[sj][threadIdx.x] = row[i];
  }

  // Fill the guard cells
  if (threadIdx.y < Pad<Order>::val) {
    Scalar* row = (Scalar*)((char*)f.ptr + offset + (threadIdx.y - Pad<Order>::val + dev_mesh.guard[2]) * dz);
    s_f[threadIdx.y][threadIdx.x] = row[i];
    row = (Scalar*)((char*)f.ptr + offset + (threadIdx.y + DIM3 + dev_mesh.guard[2]) * dz);
    s_f[DIM3 + Pad<Order>::val + threadIdx.y][threadIdx.x] = row[i];
  }

  __syncthreads();

  // compute the derivative
  for (int j = threadIdx.y; j < DIM3; j += blockDim.y) {
    int sj = j + Pad<Order>::val;
    Scalar* row_df = (Scalar*)((char*)df.ptr + (j + dev_mesh.guard[2]) * dz + offset);
    // row_df[i] += (s_f[sj + stagger][threadIdx.x] -
    //               s_f[sj + stagger - 1][threadIdx.x]) * q /
    //              dev_mesh.delta[2];
    row_df[i] += q * dy<Order>(s_f, threadIdx.x, sj + stagger, dev_mesh.delta[2]);
  }
}

template <int Order, int DIM1, int DIM2, int DIM3>
__device__ __forceinline__
void init_shared_memory(Scalar s_u1[][DIM2 + Pad<Order>::val*2][DIM1 + Pad<Order>::val*2],
                        Scalar s_u2[][DIM2 + Pad<Order>::val*2][DIM1 + Pad<Order>::val*2],
                        Scalar s_u3[][DIM2 + Pad<Order>::val*2][DIM1 + Pad<Order>::val*2],
                        cudaPitchedPtr& u1, cudaPitchedPtr& u2, cudaPitchedPtr& u3,
                        size_t globalOffset, int c1, int c2, int c3) {
  // Load field values into shared memory
  s_u1[c3][c2][c1] = *(Scalar*)((char*)u1.ptr + globalOffset);
  s_u2[c3][c2][c1] = *(Scalar*)((char*)u2.ptr + globalOffset);
  s_u3[c3][c2][c1] = *(Scalar*)((char*)u3.ptr + globalOffset);

  // Handle extra guard cells
  if (c1 < 2*Pad<Order>::val) {
    s_u1[c3][c2][c1 - Pad<Order>::val] =
        *(Scalar*)((char*)u1.ptr + globalOffset - Pad<Order>::val*sizeof(Scalar));
    s_u2[c3][c2][c1 - Pad<Order>::val] =
        *(Scalar*)((char*)u2.ptr + globalOffset - Pad<Order>::val*sizeof(Scalar));
    s_u3[c3][c2][c1 - Pad<Order>::val] =
        *(Scalar*)((char*)u3.ptr + globalOffset - Pad<Order>::val*sizeof(Scalar));
    s_u1[c3][c2][c1 + DIM1] =
        *(Scalar*)((char*)u1.ptr + globalOffset + DIM1*sizeof(Scalar));
    s_u2[c3][c2][c1 + DIM1] =
        *(Scalar*)((char*)u2.ptr + globalOffset + DIM1*sizeof(Scalar));
    s_u3[c3][c2][c1 + DIM1] =
        *(Scalar*)((char*)u3.ptr + globalOffset + DIM1*sizeof(Scalar));
  }
  if (c2 < 2*Pad<Order>::val) {
    s_u1[c3][c2 - Pad<Order>::val][c1] =
        *(Scalar*)((char*)u1.ptr + globalOffset - Pad<Order>::val*u1.pitch);
    s_u2[c3][c2 - Pad<Order>::val][c1] =
        *(Scalar*)((char*)u2.ptr + globalOffset - Pad<Order>::val*u2.pitch);
    s_u3[c3][c2 - Pad<Order>::val][c1] =
        *(Scalar*)((char*)u3.ptr + globalOffset - Pad<Order>::val*u3.pitch);
    s_u1[c3][c2 + DIM2][c1] =
        *(Scalar*)((char*)u1.ptr + globalOffset + DIM2*u1.pitch);
    s_u2[c3][c2 + DIM2][c1] =
        *(Scalar*)((char*)u2.ptr + globalOffset + DIM2*u2.pitch);
    s_u3[c3][c2 + DIM2][c1] =
        *(Scalar*)((char*)u3.ptr + globalOffset + DIM2*u3.pitch);
  }
  if (c3 < 2*Pad<Order>::val) {
    s_u1[c3 - Pad<Order>::val][c2][c1] =
        *(Scalar*)((char*)u1.ptr + globalOffset - Pad<Order>::val*u1.pitch * u1.ysize);
    s_u2[c3 - Pad<Order>::val][c2][c1] =
        *(Scalar*)((char*)u2.ptr + globalOffset - Pad<Order>::val*u2.pitch * u2.ysize);
    s_u3[c3 - Pad<Order>::val][c2][c1] =
        *(Scalar*)((char*)u3.ptr + globalOffset - Pad<Order>::val*u3.pitch * u3.ysize);
    s_u1[c3 + DIM3][c2][c1] =
        *(Scalar*)((char*)u1.ptr + globalOffset + DIM3*u1.pitch * u1.ysize);
    s_u2[c3 + DIM3][c2][c1] =
        *(Scalar*)((char*)u2.ptr + globalOffset + DIM3*u2.pitch * u2.ysize);
    s_u3[c3 + DIM3][c2][c1] =
        *(Scalar*)((char*)u3.ptr + globalOffset + DIM3*u3.pitch * u3.ysize);
  }
}

template <int Order, int DIM1, int DIM2, int DIM3>
__device__ __forceinline__
void init_shared_memory(Scalar s_f[][DIM2 + Pad<Order>::val*2][DIM1 + Pad<Order>::val*2],
                        cudaPitchedPtr& f, size_t globalOffset, int c1, int c2, int c3) {
  // Load field values into shared memory
  s_f[c3][c2][c1] = *(Scalar*)((char*)f.ptr + globalOffset);

  // Handle extra guard cells
  if (c1 < 2*Pad<Order>::val) {
    s_f[c3][c2][c1 - Pad<Order>::val] =
        *(Scalar*)((char*)f.ptr + globalOffset - Pad<Order>::val*sizeof(Scalar));
    s_f[c3][c2][c1 + DIM1] =
        *(Scalar*)((char*)f.ptr + globalOffset + DIM1*sizeof(Scalar));
  }
  if (c2 < 2*Pad<Order>::val) {
    s_f[c3][c2 - Pad<Order>::val][c1] =
        *(Scalar*)((char*)f.ptr + globalOffset - Pad<Order>::val*f.pitch);
    s_f[c3][c2 + DIM2][c1] =
        *(Scalar*)((char*)f.ptr + globalOffset + DIM2*f.pitch);
  }
  if (c3 < 2*Pad<Order>::val) {
    s_f[c3 - Pad<Order>::val][c2][c1] =
        *(Scalar*)((char*)f.ptr + globalOffset - Pad<Order>::val*f.pitch * f.ysize);
    s_f[c3 + DIM3][c2][c1] =
        *(Scalar*)((char*)f.ptr + globalOffset + DIM3*f.pitch * f.ysize);
  }
}

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


template <int Order, int DIM1, int DIM2, int DIM3>
__global__
void compute_curl(cudaPitchedPtr v1, cudaPitchedPtr v2, cudaPitchedPtr v3,
                  cudaPitchedPtr u1, cudaPitchedPtr u2, cudaPitchedPtr u3,
                  Stagger s1, Stagger s2, Stagger s3, Scalar q = 1.0) {
  // Declare cache array in shared memory
  __shared__ Scalar s_u1[DIM3 + 2*Pad<Order>::val]
      [DIM2 + 2*Pad<Order>::val][DIM1 + 2*Pad<Order>::val];
  __shared__ Scalar s_u2[DIM3 + 2*Pad<Order>::val]
      [DIM2 + 2*Pad<Order>::val][DIM1 + 2*Pad<Order>::val];
  __shared__ Scalar s_u3[DIM3 + 2*Pad<Order>::val]
      [DIM2 + 2*Pad<Order>::val][DIM1 + 2*Pad<Order>::val];

  // Load shared memory
  int t1 = blockIdx.x, t2 = blockIdx.y, t3 = blockIdx.z;
  int c1 = threadIdx.x + Pad<Order>::val,
      c2 = threadIdx.y + Pad<Order>::val,
      c3 = threadIdx.z + Pad<Order>::val;
  size_t globalOffset =  (dev_mesh.guard[2] + t3 * DIM3 + c3 - Pad<Order>::val) * u1.pitch * u1.ysize +
                         (dev_mesh.guard[1] + t2 * DIM2 + c2 - Pad<Order>::val) * u1.pitch +
                         (dev_mesh.guard[0] + t1 * DIM1 + c1 - Pad<Order>::val) * sizeof(Scalar);

  init_shared_memory<Order, DIM1, DIM2, DIM3>(s_u1, s_u2, s_u3, u1, u2, u3,
                                       globalOffset, c1, c2, c3);
  __syncthreads();

  // Do the actual computation here
  // (Curl u)_1 = d2u3 - d3u2
  (*(Scalar*)((char*)v1.ptr + globalOffset)) =
      d2<Order, DIM1, DIM2, DIM3>(s_u3, c1, c2 + flip(s3[1]), c3) -
      d3<Order, DIM1, DIM2, DIM3>(s_u2, c1, c2, c3 + flip(s2[2]));

  // (Curl u)_2 = d3u1 - d1u3
  (*(Scalar*)((char*)v2.ptr + globalOffset)) =
      d3<Order, DIM1, DIM2, DIM3>(s_u1, c1, c2, c3 + flip(s1[2])) -
      d1<Order, DIM1, DIM2, DIM3>(s_u3, c1 + flip(s3[0]), c2, c3);

  // (Curl u)_3 = d1u2 - d2u1
  (*(Scalar*)((char*)v3.ptr + globalOffset)) =
      d1<Order, DIM1, DIM2, DIM3>(s_u2, c1 + flip(s2[0]), c2, c3) -
      d2<Order, DIM1, DIM2, DIM3>(s_u1, c1, c2 + flip(s1[1]), c3);
  //     }
  //   }
  // }
}

template <int Order, int DIM1, int DIM2, int DIM3>
__global__
void compute_div(cudaPitchedPtr v, cudaPitchedPtr u1,
                 cudaPitchedPtr u2, cudaPitchedPtr u3,
                 Stagger s1, Stagger s2, Stagger s3, Scalar q = 1.0) {
  // Declare cache array in shared memory
  __shared__ Scalar s_u1[DIM3 + 2*Pad<Order>::val]
      [DIM2 + 2*Pad<Order>::val][DIM1 + 2*Pad<Order>::val];
  __shared__ Scalar s_u2[DIM3 + 2*Pad<Order>::val]
      [DIM2 + 2*Pad<Order>::val][DIM1 + 2*Pad<Order>::val];
  __shared__ Scalar s_u3[DIM3 + 2*Pad<Order>::val]
      [DIM2 + 2*Pad<Order>::val][DIM1 + 2*Pad<Order>::val];

  // Load indices
  int t1 = blockIdx.x, t2 = blockIdx.y, t3 = blockIdx.z;
  int c1 = threadIdx.x + Pad<Order>::val,
      c2 = threadIdx.y + Pad<Order>::val,
      c3 = threadIdx.z + Pad<Order>::val;
  size_t globalOffset =  + (dev_mesh.guard[2] + t3 * DIM3 + c3 - Pad<Order>::val) * u1.pitch * u1.ysize
                         + (dev_mesh.guard[1] + t2 * DIM2 + c2 - Pad<Order>::val) * u1.pitch
                         + (dev_mesh.guard[0] + t1 * DIM1 + c1 - Pad<Order>::val) * sizeof(Scalar);

  init_shared_memory<Order, DIM1, DIM2, DIM3>(s_u1, s_u2, s_u3, u1, u2, u3,
                                       globalOffset, c1, c2, c3);
  __syncthreads();

  // Do the actual computation here
  (*(Scalar*)((char*)v.ptr + globalOffset)) =
      q * d1<Order, DIM1, DIM2, DIM3>(s_u1, c1 + flip(s1[0]), c2, c3) +
      q * d2<Order, DIM1, DIM2, DIM3>(s_u2, c1, c2 + flip(s2[1]), c3) +
      q * d3<Order, DIM1, DIM2, DIM3>(s_u3, c1, c2, c3 + flip(s3[2]));
}

template <int Order, int DIM1, int DIM2, int DIM3>
__global__
void compute_grad(cudaPitchedPtr v1, cudaPitchedPtr v2, cudaPitchedPtr v3,
                  cudaPitchedPtr u, Stagger s, Scalar q = 1.0) {
  // Declare cache array in shared memory
  __shared__ Scalar s_u[DIM3 + 2*Pad<Order>::val]
      [DIM2 + 2*Pad<Order>::val][DIM1 + 2*Pad<Order>::val];

  // Load indices
  int t1 = blockIdx.x, t2 = blockIdx.y, t3 = blockIdx.z;
  int c1 = threadIdx.x + Pad<Order>::val,
      c2 = threadIdx.y + Pad<Order>::val,
      c3 = threadIdx.z + Pad<Order>::val;
  size_t globalOffset =  + (dev_mesh.guard[2] + t3 * DIM3 + c3 - Pad<Order>::val) * u.pitch * u.ysize
                         + (dev_mesh.guard[1] + t2 * DIM2 + c2 - Pad<Order>::val) * u.pitch
                         + (dev_mesh.guard[0] + t1 * DIM1 + c1 - Pad<Order>::val) * sizeof(Scalar);

  // Load field values into shared memory
  init_shared_memory<Order, DIM1, DIM2, DIM3>(s_u, u,
                                              globalOffset, c1, c2, c3);
  __syncthreads();

  (*(Scalar*)((char*)v1.ptr + globalOffset)) = q * d1<2, DIM1, DIM2, DIM3>(s_u, c1 + flip(s[0]), c2, c3);
  (*(Scalar*)((char*)v2.ptr + globalOffset)) = q * d2<2, DIM1, DIM2, DIM3>(s_u, c1, c2 + flip(s[1]), c3);
  (*(Scalar*)((char*)v3.ptr + globalOffset)) = q * d3<2, DIM1, DIM2, DIM3>(s_u, c1, c2, c3 + flip(s[2]));
}

}

void curl(VectorField<Scalar>& result, const VectorField<Scalar>& u, Scalar q) {
  auto& grid = u.grid();
  auto& mesh = grid.mesh();

  dim3 blockSize(32, 8, 4);
  dim3 gridSize(mesh.reduced_dim(0) / 32, mesh.reduced_dim(1) / 8,
                mesh.reduced_dim(2) / 4);
  Kernels::compute_curl<order, 32, 8, 4><<<gridSize, blockSize>>>
      (result.ptr(0), result.ptr(1), result.ptr(2),
       u.ptr(0), u.ptr(1), u.ptr(2),
       u.stagger(0), u.stagger(1), u.stagger(2), q);
  CudaCheckError();
}

void curl_add(VectorField<Scalar>& result, const VectorField<Scalar>& u, Scalar q) {
  auto& grid = u.grid();
  auto& mesh = grid.mesh();

  // TODO: reset the result first?

  // TODO: The kernel launch parameters might need some tuning for different
  // architectures
  dim3 blockSizeX(64, 8, 1);
  dim3 gridSizeX(mesh.reduced_dim(0) / 64, mesh.reduced_dim(1) / 8,
                 mesh.reduced_dim(2));
  dim3 blockSizeY(32, 16, 1);
  dim3 gridSizeY(mesh.reduced_dim(0) / 32, mesh.reduced_dim(1) / 64,
                 mesh.reduced_dim(2));
  dim3 blockSizeZ(32, 16, 1);
  dim3 gridSizeZ(mesh.reduced_dim(0) / 32, mesh.reduced_dim(2) / 64,
                 mesh.reduced_dim(1));

  // v3 = d1u2 - d2u1
  Kernels::deriv_x<order, 64, 8><<<gridSizeX, blockSizeX>>>
      (result.ptr(2), u.ptr(1), flip(u.stagger(1)[0]), 1.0f);
  CudaCheckError();

  if (u.grid().dim() > 1) {
    Kernels::deriv_y<order, 32, 64><<<gridSizeY, blockSizeY>>>
        (result.ptr(2), u.ptr(0), flip(u.stagger(0)[1]), -1.0f);
    CudaCheckError();
  }

  // v2 = d3u1 - d1u3
  if (u.grid().dim() > 2) {
    Kernels::deriv_z<order, 32, 64><<<gridSizeZ, blockSizeZ>>>
        (result.ptr(1), u.ptr(0), flip(u.stagger(0)[2]), 1.0f);
    CudaCheckError();
  }

    Kernels::deriv_x<order, 64, 8><<<gridSizeX, blockSizeX>>>
        (result.ptr(1), u.ptr(2), flip(u.stagger(2)[0]), -1.0f);
    CudaCheckError();

  // v1 = d2u3 - d3u2
  if (u.grid().dim() > 1) {
    Kernels::deriv_y<order, 32, 64><<<gridSizeY, blockSizeY>>>
        (result.ptr(0), u.ptr(2), flip(u.stagger(2)[1]), 1.0f);
    CudaCheckError();
  }

  if (u.grid().dim() > 2) {
    Kernels::deriv_z<order, 32, 64><<<gridSizeZ, blockSizeZ>>>
        (result.ptr(0), u.ptr(1), flip(u.stagger(1)[2]), -1.0f);
    CudaCheckError();
  }
}

void div(ScalarField<Scalar>& result, const VectorField<Scalar>& u, Scalar q) {
  auto& grid = u.grid();
  auto& mesh = grid.mesh();

  // TODO: reset the result first?

  // TODO: The kernel launch parameters might need some tuning for different
  // architectures

  dim3 blockSize(32, 8, 4);
  dim3 gridSize(mesh.reduced_dim(0) / 32, mesh.reduced_dim(1) / 8,
                mesh.reduced_dim(2) / 4);
  Kernels::compute_div<order, 32, 8, 4><<<gridSize, blockSize>>>
      (result.ptr(), u.ptr(0), u.ptr(1), u.ptr(2),
       u.stagger(0), u.stagger(1), u.stagger(2), q);
  CudaCheckError();
}

void div_add(ScalarField<Scalar>& result, const VectorField<Scalar>& u, Scalar q) {
  auto& grid = u.grid();
  auto& mesh = grid.mesh();

  // TODO: reset the result first?

  // TODO: The kernel launch parameters might need some tuning for different
  // architectures
  dim3 blockSizeX(64, 8, 1);
  dim3 gridSizeX(mesh.reduced_dim(0) / 64, mesh.reduced_dim(1) / 8,
                 mesh.reduced_dim(2));
  dim3 blockSizeY(32, 16, 1);
  dim3 gridSizeY(mesh.reduced_dim(0) / 32, mesh.reduced_dim(1) / 64,
                 mesh.reduced_dim(2));
  dim3 blockSizeZ(32, 16, 1);
  dim3 gridSizeZ(mesh.reduced_dim(0) / 32, mesh.reduced_dim(2) / 64,
                 mesh.reduced_dim(1));

  Kernels::deriv_x<order, 64, 8><<<gridSizeX, blockSizeX>>>
      (result.ptr(), u.ptr(0), flip(u.stagger(0)[0]), 1.0f);
  CudaCheckError();

  if (grid.dim() > 1) {
    Kernels::deriv_y<order, 32, 64><<<gridSizeY, blockSizeY>>>
        (result.ptr(), u.ptr(1), flip(u.stagger(1)[1]), 1.0f);
    CudaCheckError();
  }

  if (grid.dim() > 2) {
    Kernels::deriv_z<order, 32, 64><<<gridSizeZ, blockSizeZ>>>
        (result.ptr(), u.ptr(2), flip(u.stagger(2)[2]), 1.0f);
    CudaCheckError();
  }
}

void grad(VectorField<Scalar>& result, const ScalarField<Scalar>& u, Scalar q) {
  auto& grid = u.grid();
  auto& mesh = grid.mesh();

  // TODO: reset the result first?

  // TODO: The kernel launch parameters might need some tuning for different
  // architectures

  dim3 blockSize(16, 8, 8);
  dim3 gridSize(mesh.reduced_dim(0) / 16, mesh.reduced_dim(1) / 8,
                mesh.reduced_dim(2) / 8);
  Kernels::compute_grad<order, 16, 8, 8><<<gridSize, blockSize>>>
      (result.ptr(0), result.ptr(1), result.ptr(2), u.ptr(),
       u.stagger(), q);
  CudaCheckError();
}

void grad_add(VectorField<Scalar>& result, const ScalarField<Scalar>& u, Scalar q) {
  auto& grid = u.grid();
  auto& mesh = grid.mesh();

  // TODO: reset the result first?

  // TODO: The kernel launch parameters might need some tuning for different
  // architectures
  dim3 blockSizeX(64, 8, 1);
  dim3 gridSizeX(mesh.reduced_dim(0) / 64, mesh.reduced_dim(1) / 8,
                 mesh.reduced_dim(2));
  dim3 blockSizeY(32, 16, 1);
  dim3 gridSizeY(mesh.reduced_dim(0) / 32, mesh.reduced_dim(1) / 64,
                 mesh.reduced_dim(2));
  dim3 blockSizeZ(32, 16, 1);
  dim3 gridSizeZ(mesh.reduced_dim(0) / 32, mesh.reduced_dim(2) / 64,
                 mesh.reduced_dim(1));

  Kernels::deriv_x<order, 64, 8><<<gridSizeX, blockSizeX>>>
      (result.ptr(0), u.ptr(), flip(u.stagger()[0]), 1.0f);
  CudaCheckError();

  if (grid.dim() > 1) {
    Kernels::deriv_y<order, 32, 64><<<gridSizeY, blockSizeY>>>
        (result.ptr(1), u.ptr(), flip(u.stagger()[1]), 1.0f);
    CudaCheckError();
  }

  if (grid.dim() > 2) {
    Kernels::deriv_z<order, 32, 64><<<gridSizeZ, blockSizeZ>>>
        (result.ptr(2), u.ptr(), flip(u.stagger()[2]), 1.0f);
    CudaCheckError();
  }
}

void ffe_edotb(ScalarField<Scalar>& result, const VectorField<Scalar>& E,
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

void ffe_j(VectorField<Scalar>& result, const ScalarField<Scalar>& tmp_f,
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
