#include "multi_array_impl.hpp"
// #include "utils/simd.h"
#include <algorithm>
#include <cstring>

namespace Aperture {

template <typename T>
void
multi_array<T>::copy_from(const self_type& other) {
  if (m_size != other.m_size) {
    throw std::range_error(
        "Trying to copy from a multi_array of different size!");
  }
  memcpy(m_data_h, other.m_data_h, m_size * sizeof(T));
}

template <typename T>
void
multi_array<T>::alloc_mem(const Extent& ext) {
  if (m_data_h != nullptr)
    free_mem();

  auto size = ext.size();
  m_data_h = new T[size];
}

template <typename T>
void
multi_array<T>::free_mem() {
  if (m_data_h != nullptr) {
    delete[] m_data_h;
    m_data_h = nullptr;
  }
}

template <typename T>
void
multi_array<T>::assign_dev(const T& value) {}

template <typename T>
void
multi_array<T>::sync_to_host() {}

template <typename T>
void
multi_array<T>::sync_to_device() {}

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
// template class multi_array<simd::simd_buffer>;

}  // namespace Aperture
