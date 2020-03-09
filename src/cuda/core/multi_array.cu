#include "core/multi_array_impl.hpp"
#include "cuda/cudaUtility.h"
#include "cuda/data/multi_array_utils.cuh"
#include "cuda/utils/interpolation.cuh"
#include "cuda/utils/pitchptr.h"
#include "utils/logger.h"
#include <algorithm>
#include <cstring>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>

namespace Aperture {

namespace Kernels {

template <typename T>
__global__ void
downsample(pitchptr<T> orig_data, pitchptr<float> dst_data,
           Extent orig_ext, Extent dst_ext, Index offset, Stagger st,
           int d) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int k = threadIdx.z + blockIdx.z * blockDim.z;
  if (i < dst_ext.x && j < dst_ext.y && k < dst_ext.z) {
    size_t orig_idx =
        (i * d + offset.x) * sizeof(T) +
        (j * d + offset.y) * orig_data.p.pitch +
        (k * d + offset.z) * orig_data.p.pitch * orig_data.p.ysize;
    size_t dst_idx = i * sizeof(T) + j * dst_data.p.pitch +
                     k * dst_data.p.pitch * dst_data.p.ysize;

    dst_data[dst_idx] =
        interpolate(orig_data, orig_idx, st, Stagger(0b111),
                    orig_data.p.pitch, orig_data.p.ysize);
  }
}

template <typename T>
__global__ void
downsample2d(pitchptr<T> orig_data, pitchptr<float> dst_data,
             Extent orig_ext, Extent dst_ext, Index offset, Stagger st,
             int d) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if (i < dst_ext.x && j < dst_ext.y) {
    size_t orig_idx = (i * d + offset.x) * sizeof(T) +
                      (j * d + offset.y) * orig_data.p.pitch;
    size_t dst_idx = i * sizeof(T) + j * dst_data.p.pitch;

    dst_data[dst_idx] = interpolate2d(
        orig_data, orig_idx, st, Stagger(0b111), orig_data.p.pitch);
  }
}

template <typename T>
__global__ void
downsample1d(pitchptr<T> orig_data, pitchptr<float> dst_data,
             Extent orig_ext, Extent dst_ext, Index offset, Stagger st,
             int d) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < dst_ext.x) {
    size_t orig_idx = (i * d + offset.x) * sizeof(T);
    size_t dst_idx = i * sizeof(T);

    dst_data[dst_idx] =
        interpolate1d(orig_data, orig_idx, st, Stagger(0b111));
  }
}

}  // namespace Kernels

template <typename T>
inline cudaExtent
cuda_ext(const Extent& ext, const T& t) {
  return make_cudaExtent(ext.x * sizeof(T), ext.y, ext.z);
}

template <typename T>
void
multi_array<T>::copy_from(const self_type& other) {
  if (m_size != other.m_size) {
    throw std::range_error(
        "Trying to copy from a multi_array of different size!");
  }
  // memcpy(m_data_h, other.m_data_h, m_size * sizeof(T));
  assert(m_extent == other.m_extent);
  cudaMemcpy3DParms myParms = {};

  myParms.srcPtr = make_cudaPitchedPtr(other.m_data_d, other.m_pitch,
                                       other.m_extent.width(),
                                       other.m_extent.height());
  myParms.srcPos = make_cudaPos(0, 0, 0);
  myParms.dstPtr = make_cudaPitchedPtr(
      m_data_d, m_pitch, m_extent.width(), m_extent.height());
  myParms.dstPos = make_cudaPos(0, 0, 0);
  myParms.extent = cuda_ext(m_extent, T{});
  myParms.kind = cudaMemcpyDeviceToDevice;

  CudaSafeCall(cudaMemcpy3D(&myParms));
  // copy_to_host();
}

template <typename T>
void
multi_array<T>::copy_from(self_type& other, const Index& idx_src,
                          const Index& idx_dst, const Extent& ext,
                          int type) {
  // Get rid of default
  if (type == (int)cudaMemcpyDefault)
    type = (int)cudaMemcpyDeviceToDevice;

  check_dimensions(other, idx_src, idx_dst, ext);
  cudaExtent cu_ext = cuda_ext(ext, T{});
  // make_cudaExtent(ext.x * sizeof(T), ext.y, ext.z);

  // Here we use the convention of:
  // cudaMemcpyHostToHost = 0
  // cudaMemcpyHostToDevice = 1
  // cudaMemcpyDeviceToHost = 2
  // cudaMemcpyDeviceToDevice = 3
  cudaMemcpy3DParms copy_parms = {0};
  if (type < 2) {  // Copying from host
    copy_parms.srcPtr = make_cudaPitchedPtr(
        other.host_ptr(), other.m_extent.x * sizeof(T),
        other.m_extent.x, other.m_extent.y);
  } else {  // Copying from dev
    copy_parms.srcPtr =
        make_cudaPitchedPtr(other.dev_ptr(), other.m_pitch,
                            other.m_extent.x, other.m_extent.y);
  }
  copy_parms.srcPos =
      make_cudaPos(idx_src.x * sizeof(T), idx_src.y, idx_src.z);
  if (type % 2 == 0) {  // Copying to host
    copy_parms.dstPtr = make_cudaPitchedPtr(
        m_data_h, m_extent.x * sizeof(T), m_extent.x, m_extent.y);
  } else {  // Copying to device
    copy_parms.dstPtr =
        make_cudaPitchedPtr(m_data_d, m_pitch, m_extent.x, m_extent.y);
  }
  copy_parms.dstPos =
      make_cudaPos(idx_dst.x * sizeof(T), idx_dst.y, idx_dst.z);
  copy_parms.extent = cu_ext;
  copy_parms.kind = (cudaMemcpyKind)type;
  CudaSafeCall(cudaMemcpy3D(&copy_parms));
}

template <typename T>
void
multi_array<T>::add_from(self_type& other, const Index& idx_src,
                         const Index& idx_dst, const Extent& ext) {
  check_dimensions(other, idx_src, idx_dst, ext);

  // By default, this only carries out the addition on device, so we
  // first need to copy to device
  // other.copy_to_device();

  if (dim() == 3) {
    dim3 gridSize(8, 16, 16);
    dim3 blockSize(32, 8, 4);
    Kernels::map_array_binary_op<T><<<gridSize, blockSize>>>(
        get_cudaPitchedPtr(other), get_cudaPitchedPtr(*this), idx_src,
        idx_dst, ext, detail::Op_PlusAssign<T>{});
  } else if (dim() == 2) {
    dim3 gridSize(16, 32);
    dim3 blockSize(32, 8);
    Kernels::map_array_binary_op_2d<T><<<gridSize, blockSize>>>(
        get_cudaPitchedPtr(other), get_cudaPitchedPtr(*this), idx_src,
        idx_dst, ext, detail::Op_PlusAssign<T>{});
  } else {
    Kernels::map_array_binary_op_1d<T><<<128, 512>>>(
        get_cudaPitchedPtr(other), get_cudaPitchedPtr(*this), idx_src,
        idx_dst, ext, detail::Op_PlusAssign<T>{});
  }
  CudaSafeCall(cudaDeviceSynchronize());
  CudaCheckError();
}

template <typename T>
void
multi_array<T>::alloc_mem(const Extent& ext) {
  if (m_data_h != nullptr || m_data_d != nullptr) free_mem();
  auto size = ext.size();
  m_data_h = new T[size];

  auto extent = cuda_ext(ext, T{});
  cudaPitchedPtr ptr;
  CudaSafeCall(cudaMalloc3D(&ptr, extent));
  m_data_d = ptr.ptr;
  m_pitch = ptr.pitch;
  // CudaSafeCall(cudaMalloc(&m_data_d, size * sizeof(T)));
  // m_pitch = ext.x * sizeof(T);
  // cudaPitchedPtr ptr = make_cudaPitchedPtr(m_data_d, m_pitch, ext.x,
  // ext.y);
  Logger::print_info("pitch is {}, x is {}, y is {}, z is {}", m_pitch,
                     ptr.xsize, ptr.ysize, ext.z);
  Logger::print_info("--- Allocated {} bytes",
                     m_pitch * ptr.ysize * ext.z);
}

template <typename T>
void
multi_array<T>::free_mem() {
  if (m_data_h != nullptr) {
    delete[] m_data_h;
    m_data_h = nullptr;
  }
  if (m_data_d != nullptr) {
    CudaSafeCall(cudaFree(m_data_d));
    m_data_d = nullptr;
  }
}

template <typename T>
void
multi_array<T>::assign_dev(const T& value) {
  cudaPitchedPtr p = get_cudaPitchedPtr(*this);
  if (m_extent.depth() > 1) {
    // Logger::print_info("assign_dev 3d version");
    dim3 blockSize(8, 8, 8);
    // dim3 gridSize(8, 8, 8);
    dim3 gridSize((m_extent.x + 7) / 8, (m_extent.y + 7) / 8,
                  (m_extent.z + 7) / 8);
    Kernels::map_array_unary_op<T><<<gridSize, blockSize>>>(
        p, m_extent, detail::Op_AssignConst<T>(value));
    CudaCheckError();
  } else if (m_extent.height() > 1) {
    // Logger::print_info("assign_dev 2d version");
    dim3 blockSize(32, 16);
    dim3 gridSize((m_extent.x + 31) / 32, (m_extent.y + 15) / 16);
    Kernels::map_array_unary_op_2d<T><<<gridSize, blockSize>>>(
        p, m_extent, detail::Op_AssignConst<T>(value));
    CudaCheckError();
  } else if (m_extent.width() > 1) {
    Kernels::map_array_unary_op_1d<T>
        <<<64, 128>>>(p, m_extent, detail::Op_AssignConst<T>(value));
    CudaCheckError();
  }
}

template <typename T>
void
multi_array<T>::copy_to_host() {
  cudaMemcpy3DParms myParms = {};
  myParms.srcPtr = make_cudaPitchedPtr(
      m_data_d, m_pitch, m_extent.width(), m_extent.height());
  myParms.srcPos = make_cudaPos(0, 0, 0);
  myParms.dstPtr =
      make_cudaPitchedPtr(m_data_h, sizeof(T) * m_extent.width(),
                          m_extent.width(), m_extent.height());
  myParms.dstPos = make_cudaPos(0, 0, 0);
  myParms.extent = cuda_ext(m_extent, T{});
  myParms.kind = cudaMemcpyDeviceToHost;

  // Logger::print_info("copy to host with dev pitch {}, host pitch {}",
  // m_pitch,
  //                    sizeof(T) * m_extent.width());
  CudaSafeCall(cudaMemcpy3D(&myParms));
}

template <typename T>
void
multi_array<T>::copy_to_device() {
  cudaMemcpy3DParms myParms = {};
  myParms.srcPtr =
      make_cudaPitchedPtr(m_data_h, sizeof(T) * m_extent.width(),
                          m_extent.width(), m_extent.height());
  myParms.srcPos = make_cudaPos(0, 0, 0);
  myParms.dstPtr = make_cudaPitchedPtr(
      m_data_d, m_pitch, m_extent.width(), m_extent.height());
  myParms.dstPos = make_cudaPos(0, 0, 0);
  myParms.extent = cuda_ext(m_extent, T{});
  myParms.kind = cudaMemcpyHostToDevice;

  CudaSafeCall(cudaMemcpy3D(&myParms));
}

template <typename T>
void
multi_array<T>::downsample(int d, multi_array<float>& array,
                           Index offset, Stagger stagger) {
  auto& ext = array.extent();
  if (ext.y == 1 && ext.z == 1) {
    int blockSize = 512;
    int gridSize = (blockSize + ext.x - 1) / blockSize;
    Kernels::downsample1d<<<gridSize, blockSize>>>(
        get_pitchptr(*this), get_pitchptr(array), m_extent,
        array.extent(), offset, stagger, d);
    CudaCheckError();
  } else if (ext.z == 1) {  // Use 2D version
    dim3 blockSize(32, 16);
    dim3 gridSize((ext.x + blockSize.x - 1) / blockSize.x,
                  (ext.y + blockSize.y - 1) / blockSize.y);
    Kernels::downsample2d<<<gridSize, blockSize>>>(
        get_pitchptr(*this), get_pitchptr(array), m_extent,
        array.extent(), offset, stagger, d);
    CudaCheckError();
  } else {
    dim3 blockSize(32, 8, 4);
    dim3 gridSize((ext.x + blockSize.x - 1) / blockSize.x,
                  (ext.y + blockSize.y - 1) / blockSize.y,
                  (ext.z + blockSize.z - 1) / blockSize.z);
    Kernels::downsample<<<gridSize, blockSize>>>(
        get_pitchptr(*this), get_pitchptr(array), m_extent,
        array.extent(), offset, stagger, d);
    CudaCheckError();
  }
  CudaSafeCall(cudaDeviceSynchronize());
  array.copy_to_host();
}

/////////////////////////////////////////////////////////////////
// Explicitly instantiate the classes we will use
/////////////////////////////////////////////////////////////////
template class multi_array<long long>;
template class multi_array<long>;
template class multi_array<int>;
template class multi_array<short>;
template class multi_array<char>;
template class multi_array<unsigned int>;
template class multi_array<unsigned long>;
template class multi_array<unsigned long long>;
template class multi_array<float>;
template class multi_array<double>;

}  // namespace Aperture
