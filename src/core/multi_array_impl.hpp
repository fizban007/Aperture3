#ifndef __MULTI_ARRAY_IMPL_H_
#define __MULTI_ARRAY_IMPL_H_

#include "multi_array.h"

namespace Aperture {

template <typename T>
multi_array<T>::multi_array()
    : m_data_h(nullptr),
      m_data_d(nullptr),
      m_extent(0, 1, 1),
      m_size(0),
      m_pitch(0) {}

template <typename T>
multi_array<T>::multi_array(int width, int height, int depth)
    : m_data_h(nullptr),
      m_data_d(nullptr),
      m_extent(width, height, depth),
      m_pitch(0) {
  m_size = width * height * depth;

  alloc_mem(m_extent);
}

template <typename T>
multi_array<T>::multi_array(const Extent& extent)
    : multi_array(extent.width(), extent.height(), extent.depth()) {}

template <typename T>
multi_array<T>::multi_array(const self_type& other)
    : multi_array(other.m_extent) {
  copy_from(other);
}

template <typename T>
multi_array<T>::multi_array(self_type&& other) {
  m_extent = other.m_extent;
  m_size = other.m_size;
  m_data_h = other.m_data_h;
  m_data_d = other.m_data_d;

  other.m_data_h = nullptr;
  other.m_data_d = nullptr;
}

template <typename T>
multi_array<T>::~multi_array() {
  free_mem();
}

template <typename T>
multi_array<T>&
multi_array<T>::operator=(const self_type& other) {
  free_mem();
  m_size = other.m_size;
  m_extent = other.m_extent;

  alloc_mem(other.m_extent);
  copy_from(other);
  return *this;
}

template <typename T>
multi_array<T>&
multi_array<T>::operator=(self_type&& other) {
  m_data_h = other.m_data_h;
  m_data_d = other.m_data_d;

  other.m_data_h = nullptr;
  other.m_data_d = nullptr;

  m_extent = other.m_extent;
  m_size = other.m_size;

  return *this;
}

template <typename T>
const T&
multi_array<T>::operator()(int x, int y, int z) const {
  size_t idx = x + (y + z * m_extent.height()) * m_extent.width();
  return m_data_h[idx];
}

template <typename T>
T&
multi_array<T>::operator()(int x, int y, int z) {
  size_t idx = x + (y + z * m_extent.height()) * m_extent.width();
  return m_data_h[idx];
}

template <typename T>
const T&
multi_array<T>::operator()(const Index& index) const {
  return operator()(index.x, index.y, index.z);
}

template <typename T>
T&
multi_array<T>::operator()(const Index& index) {
  return operator()(index.x, index.y, index.z);
}

template <typename T>
const T& multi_array<T>::operator[](size_t n) const {
  return m_data_h[n];
}

template <typename T>
T& multi_array<T>::operator[](size_t n) {
  return m_data_h[n];
}

template <typename T>
void
multi_array<T>::assign(const T& value) {
  std::fill_n(m_data_h, m_size, value);
}

template <typename T>
void
multi_array<T>::resize(int width, int height, int depth) {
  size_t size = width * height * depth;
  m_extent = Extent(width, height, depth);
  m_size = size;

  free_mem();
  alloc_mem(m_extent);
}

template <typename T>
void
multi_array<T>::resize(Extent extent) {
  resize(extent.width(), extent.height(), extent.depth());
}

}  // namespace Aperture

#endif  // __MULTI_ARRAY_IMPL_H_
