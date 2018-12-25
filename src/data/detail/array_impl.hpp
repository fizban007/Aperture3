#ifndef _ARRAY_IMPL_H_
#define _ARRAY_IMPL_H_

#include "cuda/cudaUtility.h"
#include "cuda_runtime.h"
#include "data/array.h"
#include "utils/logger.h"
#include <algorithm>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>

namespace Aperture {

template <typename T>
Array<T>::Array() {}

template <typename T>
Array<T>::Array(size_t length, int devId) {
  alloc_mem(length, devId);
}

template <typename T>
Array<T>::Array(const self_type& other) {
  alloc_mem(other.m_length, other.m_devId);
}

template <typename T>
Array<T>::Array(self_type&& other) {
  m_data_d = other.m_data_d;
  m_data_h = other.m_data_h;
  m_length = other.m_length;
  m_devId = other.m_devId;
  // Need to set nullptr or the pointers will be freed
  other.m_data_h = nullptr;
  other.m_data_d = nullptr;
}

template <typename T>
Array<T>::~Array() {
  free_mem();
}

template <typename T>
void
Array<T>::alloc_mem(size_t N, int deviceId) {
  if (m_data_d != nullptr || m_data_h != nullptr) free_mem();
  m_devId = deviceId;
  CudaSafeCall(cudaSetDevice(m_devId));
  CudaSafeCall(cudaMalloc(&m_data_d, N * sizeof(T)));
  m_data_h = new T[N];
  m_length = N;
}

template <typename T>
void
Array<T>::free_mem() {
  CudaSafeCall(cudaSetDevice(m_devId));
  if (m_data_d != nullptr) {
    CudaSafeCall(cudaFree(m_data_d));
    m_data_d = nullptr;
  }
  if (m_data_h != nullptr) {
    delete[] m_data_h;
    m_data_h = nullptr;
  }
}

/// Sync the content between host and device
template <typename T>
void
Array<T>::sync_to_device(int devId) {
  CudaSafeCall(cudaSetDevice(devId));
  CudaSafeCall(cudaMemcpy(m_data_d, m_data_h, m_length * sizeof(T),
                          cudaMemcpyHostToDevice));
}

/// Sync the content between host and device
template <typename T>
void
Array<T>::sync_to_device() {
  sync_to_device(m_devId);
}

/// Sync the content between host and device
template <typename T>
void
Array<T>::sync_to_host() {
  CudaSafeCall(cudaSetDevice(m_devId));
  CudaSafeCall(cudaMemcpy(m_data_h, m_data_d, m_length * sizeof(T),
                          cudaMemcpyDeviceToHost));
}

/// Set part of the array to a single initial value on the host
template <typename T>
void
Array<T>::assign(const data_type& value, size_t num) {
  if (num > m_length) num = m_length;
  std::fill_n(m_data_h, num, value);
}

/// Set part of the array to a single initial value through device
/// kernel
template <typename T>
void
Array<T>::assign_dev(const data_type& value, size_t num) {
  if (num > m_length) num = m_length;
  thrust::device_ptr<T> ptr = thrust::device_pointer_cast(m_data_d);
  thrust::fill_n(ptr, num, value);
  CudaCheckError();
}

/// Set the whole array to a single initial value on the host
template <typename T>
void
Array<T>::assign(const data_type& value) {
  assign(value, m_length);
}

/// Set the whole array to a single initial value through device kernel
template <typename T>
void
Array<T>::assign_dev(const data_type& value) {
  assign_dev(value, m_length);
}

/// Resize the array.
template <typename T>
void
Array<T>::resize(size_t length, int deviceId) {
  free_mem();
  alloc_mem(length, deviceId);
}

}  // namespace Aperture

#endif  // _ARRAY_IMPL_H_
