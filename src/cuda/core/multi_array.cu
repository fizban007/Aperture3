#include "core/multi_array_impl.hpp"
#include "cuda/cudaUtility.h"
#include "cuda/data/multi_array_utils.cuh"
#include "cuda/utils/pitchptr.cuh"
#include "utils/logger.h"
#include <algorithm>
#include <cstring>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>

namespace Aperture {

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
  // sync_to_host();
}

template <typename T>
void
multi_array<T>::copy_from(self_type& other, const Index& idx_src,
                          const Index& idx_dst, const Extent& ext,
                          int type) {
  // Get rid of default
  if (type == (int)cudaMemcpyDefault)
    type = (int)cudaMemcpyDeviceToDevice;

  // Check dimensions
  if (idx_src.x + ext.x > other.m_extent.x ||
      idx_src.y + ext.y > other.m_extent.y ||
      idx_src.z + ext.z > other.m_extent.z) {
    throw std::range_error("Source extent is too large!");
  }
  if (idx_dst.x + ext.x > m_extent.x ||
      idx_dst.y + ext.y > m_extent.y ||
      idx_dst.z + ext.z > m_extent.z) {
    throw std::range_error("Destination extent is too large!");
  }

  // Here we use the convention of:
  // cudaMemcpyHostToHost = 0
  // cudaMemcpyHostToDevice = 1
  // cudaMemcpyDeviceToHost = 2
  // cudaMemcpyDeviceToDevice = 3

  cudaExtent cuda_ext =
      make_cudaExtent(ext.x * sizeof(T), ext.y, ext.z);

  cudaMemcpy3DParms copy_parms = {0};
  if (type < 2) { // Copying from host
    copy_parms.srcPtr = make_cudaPitchedPtr(
        other.host_ptr(), other.m_extent.x * sizeof(T), other.m_extent.x,
        other.m_extent.y);
  } else { // Copying from dev
    copy_parms.srcPtr = make_cudaPitchedPtr(
        other.dev_ptr(), other.m_pitch, other.m_extent.x,
        other.m_extent.y);
  }
  copy_parms.srcPos =
      make_cudaPos(idx_src.x * sizeof(T), idx_src.y, idx_src.z);
  if (type % 2 == 0) { // Copying to host
    copy_parms.dstPtr = make_cudaPitchedPtr(
        m_data_h, m_extent.x * sizeof(T), m_extent.x, m_extent.y);
  } else { // Copying to device
    copy_parms.dstPtr = make_cudaPitchedPtr(
        m_data_d, m_pitch, m_extent.x, m_extent.y);
  }
  copy_parms.dstPos =
      make_cudaPos(idx_dst.x * sizeof(T), idx_dst.y, idx_dst.z);
  copy_parms.extent = cuda_ext;
  copy_parms.kind = (cudaMemcpyKind)type;
  cudaMemcpy3D(&copy_parms);
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
  Logger::print_info("pitch is {}, x is {}, y is {}", m_pitch,
                     ptr.xsize, ptr.ysize);
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
    dim3 gridSize((this->m_extent.x + 7) / 8,
                  (this->m_extent.y + 7) / 8,
                  (this->m_extent.z + 7) / 8);
    Kernels::map_array_unary_op<T><<<gridSize, blockSize>>>(
        p, this->m_extent, detail::Op_AssignConst<T>(value));
    CudaCheckError();
  } else if (m_extent.height() > 1) {
    // Logger::print_info("assign_dev 2d version");
    dim3 blockSize(32, 16);
    dim3 gridSize((this->m_extent.x + 31) / 32,
                  (this->m_extent.y + 15) / 16);
    Kernels::map_array_unary_op_2d<T><<<gridSize, blockSize>>>(
        p, this->m_extent, detail::Op_AssignConst<T>(value));
    CudaCheckError();
  } else if (m_extent.width() > 1) {
    Kernels::map_array_unary_op_1d<T><<<64, 128>>>(
        p, this->m_extent, detail::Op_AssignConst<T>(value));
    CudaCheckError();
  }
}

template <typename T>
void
multi_array<T>::sync_to_host() {
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

  // Logger::print_info("dev pitch {}, host pitch {}", m_pitch,
  //                    sizeof(T) * m_extent.width());
  CudaSafeCall(cudaMemcpy3D(&myParms));
}

template <typename T>
void
multi_array<T>::sync_to_device() {
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
template class multi_array<long double>;

}  // namespace Aperture
