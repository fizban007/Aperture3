#include "core/multi_array_impl.hpp"
#include "cuda/cudaUtility.h"
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
  cudaMemcpy3DParms myParms = {0};

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
  sync_to_host();
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
  }
}

template <typename T>
void
multi_array<T>::assign_dev(const T& value) {}

template <typename T>
void
multi_array<T>::sync_to_host() {
  cudaMemcpy3DParms myParms = {0};
  myParms.srcPtr = make_cudaPitchedPtr(m_data_d, m_pitch,
                                       m_extent.width(),
                                       m_extent.height());
  myParms.srcPos = make_cudaPos(0, 0, 0);
  myParms.dstPtr = make_cudaPitchedPtr(m_data_h, sizeof(T)*m_extent.width(),
                                       m_extent.width(),
                                       m_extent.height());
  myParms.dstPos = make_cudaPos(0, 0, 0);
  myParms.extent = cuda_ext(m_extent, T{});
  myParms.kind = cudaMemcpyDeviceToHost;

  CudaSafeCall(cudaMemcpy3D(&myParms));
}

template <typename T>
void
multi_array<T>::sync_to_device() {
  cudaMemcpy3DParms myParms = {0};
  myParms.srcPtr = make_cudaPitchedPtr(m_data_h, sizeof(T)*m_extent.width(),
                                       m_extent.width(),
                                       m_extent.height());
  myParms.srcPos = make_cudaPos(0, 0, 0);
  myParms.dstPtr = make_cudaPitchedPtr(m_data_d, m_pitch,
                                       m_extent.width(),
                                       m_extent.height());
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
