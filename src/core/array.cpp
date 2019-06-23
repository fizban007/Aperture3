#include "array_impl.hpp"

namespace Aperture {

template <typename T>
void
array<T>::alloc_mem(size_t N) {
  if (m_data_h != nullptr) free_mem();
  m_data_h = new T[N];
  m_size = N;
}

template <typename T>
void
array<T>::free_mem() {
  if (m_data_h != nullptr) {
    delete[] m_data_h;
    m_data_h = nullptr;
  }
}

/// Sync the content between host and device
template <typename T>
void
array<T>::sync_to_device() {}

/// Sync the content between host and device
template <typename T>
void
array<T>::sync_to_host() {}

template <typename T>
void
array<T>::copy_from(const self_type &other) {
  size_t n = std::min(m_size, other.m_size);
  std::copy(other.m_data_h, other.m_data_h + n, m_data_h);
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
