#ifndef _MULTI_ARRAY_UTILS_CUH_
#define _MULTI_ARRAY_UTILS_CUH_

#include "core/constant_defs.h"
#include "core/detail/op.hpp"
#include "core/vec3.h"
#include "cuda/cuda_control.h"

namespace Aperture {

namespace Kernels {

template <typename T, typename UnaryOp>
__global__ void
map_array_unary_op(cudaPitchedPtr input, cudaPitchedPtr output,
                   const Extent ext, UnaryOp op) {
  for (int k = blockIdx.z * blockDim.z + threadIdx.z; k < ext.z;
       k += blockDim.z * gridDim.z) {
    char* slice_in = ((char*)input.ptr) + k * input.pitch * ext.y;
    char* slice_out = ((char*)output.ptr) + k * output.pitch * ext.y;
    for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < ext.y;
         j += blockDim.y * gridDim.y) {
      T* row_in = (T*)(slice_in + j * input.pitch);
      T* row_out = (T*)(slice_out + j * output.pitch);
      for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < ext.x;
           i += blockDim.x * gridDim.x) {
        // size_t idx = i + j * ext.x + k * ext.x * ext.y;
        // output[idx] = op(input[idx]);
        row_out[i] = op(row_in[i]);
      }
    }
  }
}

template <typename T, typename UnaryOp>
__global__ void
map_array_unary_op_2d(cudaPitchedPtr input, cudaPitchedPtr output,
                      const Extent ext, UnaryOp op) {
  for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < ext.y;
       j += blockDim.y * gridDim.y) {
    T* row_in = (T*)((char*)input.ptr + j * input.pitch);
    T* row_out = (T*)((char*)output.ptr + j * output.pitch);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < ext.x;
         i += blockDim.x * gridDim.x) {
      row_out[i] = op(row_in[i]);
    }
  }
}

template <typename T, typename UnaryOp>
__global__ void
map_array_unary_op_1d(cudaPitchedPtr input, cudaPitchedPtr output,
                      const Extent ext, UnaryOp op) {
  T* row_in = (T*)(input.ptr);
  T* row_out = (T*)(output.ptr);
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < ext.x;
       i += blockDim.x * gridDim.x) {
    row_out[i] = op(row_in[i]);
  }
}

template <typename T, typename UnaryOp>
__global__ void
map_array_unary_op(cudaPitchedPtr array, const Extent ext, UnaryOp op) {
  for (int k = blockIdx.z * blockDim.z + threadIdx.z; k < ext.z;
       k += blockDim.z * gridDim.z) {
    char* slice = ((char*)array.ptr) + k * array.pitch * ext.y;
    for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < ext.y;
         j += blockDim.y * gridDim.y) {
      T* row = (T*)(slice + j * array.pitch);
      for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < ext.x;
           i += blockDim.x * gridDim.x) {
        // size_t idx = i + j * ext.x + k * ext.x * ext.y;
        op(row[i]);
      }
    }
  }
}

template <typename T, typename UnaryOp>
__global__ void
map_array_unary_op_2d(cudaPitchedPtr array, const Extent ext,
                      UnaryOp op) {
  for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < ext.y;
       j += blockDim.y * gridDim.y) {
    T* row = (T*)((char*)array.ptr + j * array.pitch);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < ext.x;
         i += blockDim.x * gridDim.x) {
      // size_t idx = i + j * ext.x + k * ext.x * ext.y;
      op(row[i]);
    }
  }
}

template <typename T, typename UnaryOp>
__global__ void
map_array_unary_op_1d(cudaPitchedPtr array, const Extent ext,
                      UnaryOp op) {
  T* row = (T*)(array.ptr);
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < ext.x;
       i += blockDim.x * gridDim.x) {
    // size_t idx = i + j * ext.x + k * ext.x * ext.y;
    op(row[i]);
  }
}

template <typename T, typename BinaryOp>
__global__ void
map_array_binary_op(cudaPitchedPtr input, cudaPitchedPtr output,
                    const Extent ext, BinaryOp op) {
  for (int k = blockIdx.z * blockDim.z + threadIdx.z; k < ext.z;
       k += blockDim.z * gridDim.z) {
    char* slice_in = ((char*)input.ptr) + k * input.pitch * ext.y;
    char* slice_out = ((char*)output.ptr) + k * output.pitch * ext.y;
    for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < ext.y;
         j += blockDim.y * gridDim.y) {
      T* row_in = (T*)(slice_in + j * input.pitch);
      T* row_out = (T*)(slice_out + j * output.pitch);
      for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < ext.x;
           i += blockDim.x * gridDim.x) {
        // size_t idx = i + j * ext.x + k * ext.x * ext.y;
        // op(output[idx], input[idx]);
        op(row_out[i], row_in[i]);
      }
    }
  }
}

template <typename T, typename BinaryOp>
__global__ void
map_array_binary_op_2d(cudaPitchedPtr input, cudaPitchedPtr output,
                       const Extent ext, BinaryOp op) {
  for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < ext.y;
       j += blockDim.y * gridDim.y) {
    T* row_in = (T*)((char*)input.ptr + j * input.pitch);
    T* row_out = (T*)((char*)output.ptr + j * output.pitch);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < ext.x;
         i += blockDim.x * gridDim.x) {
      // size_t idx = i + j * ext.x + k * ext.x * ext.y;
      // op(output[idx], input[idx]);
      op(row_out[i], row_in[i]);
    }
  }
}

template <typename T, typename BinaryOp>
__global__ void
map_array_binary_op(cudaPitchedPtr a, cudaPitchedPtr b,
                    cudaPitchedPtr output, const Extent ext,
                    BinaryOp op) {
  for (int k = blockIdx.z * blockDim.z + threadIdx.z; k < ext.z;
       k += blockDim.z * gridDim.z) {
    char* slice_a = ((char*)a.ptr) + k * a.pitch * ext.y;
    char* slice_b = ((char*)b.ptr) + k * b.pitch * ext.y;
    char* slice_out = ((char*)output.ptr) + k * output.pitch * ext.y;
    for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < ext.y;
         j += blockDim.y * gridDim.y) {
      T* row_a = (T*)(slice_a + j * a.pitch);
      T* row_b = (T*)(slice_b + j * b.pitch);
      T* row_out = (T*)(slice_out + j * output.pitch);
      for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < ext.x;
           i += blockDim.x * gridDim.x) {
        // size_t idx = i + j * ext.x + k * ext.x * ext.y;
        // output[idx] = op(a[idx], b[idx]);
        row_out[i] = op(row_a[i], row_b[i]);
      }
    }
  }
}

}  // namespace Kernels

}  // namespace Aperture

#endif
