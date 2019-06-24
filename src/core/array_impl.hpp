#ifndef __ARRAY_IMPL_H_
#define __ARRAY_IMPL_H_

#include "array.h"
#include "utils/logger.h"
#include <algorithm>

namespace Aperture {

template <typename T>
array<T>::array() {}

template <typename T>
array<T>::array(size_t length) {
  alloc_mem(length);
}

template <typename T>
array<T>::array(const self_type& other) {
  alloc_mem(other.m_size);
  copy_from(other);
}

template <typename T>
array<T>::array(self_type&& other) {
  m_data_d = other.m_data_d;
  m_data_h = other.m_data_h;
  m_size = other.m_size;
  // Need to set nullptr or the pointers will be freed
  other.m_data_h = nullptr;
  other.m_data_d = nullptr;
}

template <typename T>
array<T>::~array() {
  free_mem();
}

template <typename T>
array<T>&
array<T>::operator=(self_type&& other) {
  m_data_d = other.m_data_d;
  m_data_h = other.m_data_h;
  m_size = other.m_size;

  other.m_data_d = nullptr;
  other.m_data_h = nullptr;
  return *this;
}

template <typename T>
array<T>&
array<T>::operator=(const self_type& other) {
  alloc_mem(other.m_size);
  copy_from(other);
  return *this;
}

/// Set the whole array to a single initial value on the host
template <typename T>
void
array<T>::assign(const data_type& value) {
  assign(value, m_size);
}

template <typename T>
void
array<T>::assign_dev(const data_type& value) {
  assign_dev(value, m_size);
}

template <typename T>
void
array<T>::assign(const data_type &value, size_t num) {
  if (num > m_size) num = m_size;
  std::fill_n(m_data_h, num, value);
}

/// Resize the array.
template <typename T>
void
array<T>::resize(size_t size) {
  free_mem();
  alloc_mem(size);
}

}

#endif // __ARRAY_IMPL_H_
