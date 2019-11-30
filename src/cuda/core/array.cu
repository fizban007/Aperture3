#include "core/array_impl.hpp"
#include "cuda/cudaUtility.h"
#include <thrust/device_ptr.h>
#include <thrust/fill.h>

namespace Aperture {

template <typename T>
void
array<T>::alloc_mem(size_t N) {
  if (m_data_d != nullptr || m_data_h != nullptr) free_mem();
  m_data_h = new T[N];
  CudaSafeCall(cudaMalloc(&m_data_d, N * sizeof(T)));
  m_size = N;

}

template <typename T>
void
array<T>::free_mem() {
  if (m_data_h != nullptr) {
    delete[] m_data_h;
    m_data_h = nullptr;
  }
  if (m_data_d != nullptr) {
    CudaSafeCall(cudaFree(m_data_d));
  }
}

/// Sync the content between host and device
template <typename T>
void
array<T>::copy_to_device() {
  CudaSafeCall(cudaMemcpy(m_data_d, m_data_h, m_size * sizeof(T),
                          cudaMemcpyHostToDevice));
}

/// Sync the content between host and device
template <typename T>
void
array<T>::copy_to_host() {
  CudaSafeCall(cudaMemcpy(m_data_h, m_data_d, m_size * sizeof(T),
                          cudaMemcpyDeviceToHost));
}

/// Set part of the array to a single initial value on the host
template <typename T>
void
array<T>::assign_dev(const data_type& value, size_t num) {
  if (num > m_size) num = m_size;
  thrust::device_ptr<T> ptr = thrust::device_pointer_cast(m_data_d);
  thrust::fill_n(ptr, num, value);
  CudaCheckError();
}

template <typename T>
void
array<T>::copy_from(const self_type& other) {
  size_t n = std::min(m_size, other.m_size);
  // std::copy(other.m_data_h, other.m_data_h + n, m_data_h);
  CudaSafeCall(cudaMemcpy(m_data_d, other.m_data_d, n * sizeof(T), cudaMemcpyDeviceToDevice));
  // copy_to_host();
}

template class array<long long>;
template class array<long>;
template class array<int>;
template class array<short>;
template class array<char>;
template class array<uint16_t>;
template class array<uint32_t>;
template class array<uint64_t>;
template class array<float>;
template class array<double>;
template class array<long double>;

}  // namespace Aperture
