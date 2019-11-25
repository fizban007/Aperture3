#ifndef _MULTI_ARRAY_IMPL_H_
#define _MULTI_ARRAY_IMPL_H_

#include "core/multi_array.h"
#include "utils/logger.h"
#include "utils/memory.h"

namespace Aperture {

template <typename T>
multi_array<T>::multi_array() : _data(nullptr), _size(0) {
  _extent.width() = 0;
  _extent.height() = 1;
  _extent.depth() = 1;
  _pitch = 0;
  find_dim();
}

template <typename T>
multi_array<T>::multi_array(int width, int height, int depth)
    : _data(nullptr), _extent{width, height, depth} {
  // _size = _extent.size();
  // Logger::print_info("extent has {}, {}, {}", _extent.width(),
  //                    _extent.height(), _extent.depth());
  // auto ext = cuda_ext(_extent, T{});
  // Logger::print_info("now cuda extent has {}, {}, {}", ext.width,
  //                    ext.height, ext.depth);
  find_dim();
  alloc_mem(_extent);
}

template <typename T>
multi_array<T>::multi_array(const Extent& extent)
    : multi_array(extent.width(), extent.height(), extent.depth()) {}

template <typename T>
multi_array<T>::multi_array(const self_type& other)
    : _data(nullptr),
      _extent(other._extent),
      _size(other._size),
      _dim(other._dim) {
  // _data = new T[_size];
  // _data = reinterpret_cast<T*>(aligned_malloc(_size * sizeof(T),
  // 64));
  alloc_mem(_extent);
  copy_from(other);
}

template <typename T>
multi_array<T>::multi_array(self_type&& other)
    : _extent(other._extent),
      _size(other._size),
      _dim(other._dim),
      _pitch(other._pitch) {
  _data = other._data;
  other._data = nullptr;
  other._size = 0;
  other._extent.width() = 0;
  other._extent.height() = 1;
  other._extent.depth() = 1;
  other._pitch = 0;
}

template <typename T>
multi_array<T>::~multi_array() {
  free_mem();
}

template <typename T>
void
multi_array<T>::alloc_mem(const Extent& ext) {
  if (_data != nullptr) free_mem();
  // Want to align data to 64 byte boundaries
  size_t pitch = ext.width() * sizeof(T);
  if (pitch % ALIGNMENT > 0)
    pitch = (pitch / ALIGNMENT + 1) * ALIGNMENT;
  _pitch = pitch;
  // Compute new _size
  _size = _pitch * ext.height() * ext.depth();
  // Allocate new _size with alignment
  _data = aligned_malloc(_size, ALIGNMENT);
}

template <typename T>
void
multi_array<T>::free_mem() {
  aligned_free(_data);
}

template <typename T>
void
multi_array<T>::copy_from(const self_type& other) {
  assert(_size == other._size && _pitch == other._pitch);
  memcpy(_data, other._data, _size);
}

template <typename T>
void
multi_array<T>::assign(const data_type& value) {
  // TODO: Since we will be using pitched 3d arrays, we need a new
  // method to fill the array
  for (int k = 0; k < _extent.depth(); k++) {
    for (int j = 0; j < _extent.height(); j++) {
      T* row = (T*)((char*)_data + j * _pitch +
                    k * _pitch * _extent.height());
      std::fill_n(row, _extent.width(), value);
    }
  }
}

template <typename T>
auto
multi_array<T>::operator=(const self_type& other) -> self_type& {
  if (_extent != other._extent) {
    resize(other._extent);
  }
  copy_from(other);
  return (*this);
}

template <typename T>
auto
multi_array<T>::operator=(self_type&& other) -> self_type& {
  if (_extent != other._extent) {
    _extent = other._extent;
    _size = _extent.size();
    find_dim();
  }
  // If the memory is already allocated, then pointing _data to
  // another place will lead to memory leak.
  free_mem();
  _data = other._data;
  _pitch = other._pitch;
  other._data = nullptr;
  return (*this);
}

template <typename T>
void
multi_array<T>::resize(int width, int height, int depth) {
  _extent.width() = width;
  _extent.height() = height;
  _extent.depth() = depth;
  _size = _extent.size();
  find_dim();
  free_mem();
  alloc_mem(_extent);
}

template <typename T>
void
multi_array<T>::resize(Extent extent) {
  resize(extent.width(), extent.height(), extent.depth());
}

template <typename T>
void
multi_array<T>::find_dim() {
  if (_extent.height() <= 1 && _extent.depth() <= 1)
    _dim = 1;
  else if (_extent.depth() <= 1)
    _dim = 2;
  else
    _dim = 3;
}

template <typename T>
size_t
multi_array<T>::get_offset(uint32_t idx) const {
  // int tmp = idx / _extent.width();
  return (idx % _extent.width()) * sizeof(T) +
         (idx / _extent.width()) * _pitch;
}

template <typename T>
T
multi_array<T>::interpolate(uint32_t idx, Scalar x1, Scalar x2,
                            Scalar x3, Stagger stagger) const {
  size_t offset = get_offset(idx);
  size_t k_off = _pitch * _extent.height();
  Scalar nx1 = 1.0f - x1;
  Scalar nx2 = 1.0f - x2;
  Scalar nx3 = 1.0f - x3;

  auto& data = *this;
  return nx1 * nx2 * nx3 *
             data[offset - sizeof(float) - data.pitch() - k_off] +
         x1 * nx2 * nx3 * data[offset - data.pitch() - k_off] +
         nx1 * x2 * nx3 * data[offset - sizeof(float) - k_off] +
         nx1 * nx2 * x3 * data[offset - sizeof(float) - data.pitch()] +
         x1 * x2 * nx3 * data[offset - k_off] +
         x1 * nx2 * x3 * data[offset - data.pitch()] +
         nx1 * x2 * x3 * data[offset - sizeof(float)] +
         x1 * x2 * x3 * data[offset];
}

}  // namespace Aperture

#endif  // _MULTI_ARRAY_IMPL_H_
