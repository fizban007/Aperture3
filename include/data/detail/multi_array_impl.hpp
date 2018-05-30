#ifndef  _MULTI_ARRAY_IMPL_H_
#define  _MULTI_ARRAY_IMPL_H_

#include "data/multi_array.h"
#include "utils/memory.h"
#include "utils/logger.h"

namespace Aperture {

template <typename T>
MultiArray<T>::MultiArray()
    : _data(nullptr), _size(0) {
  _extent.width() = 0;
  _extent.height() = 1;
  _extent.depth() = 1;
  find_dim();
}

template <typename T>
MultiArray<T>::MultiArray(int width, int height, int depth)
    : _extent{width, height, depth} {
  _size = _extent.size();
  find_dim();

  // _data = new T[_size];
  // _data = reinterpret_cast<T*>(aligned_malloc(_size * sizeof(T), 64));
  auto error = cudaMallocManaged(&_data, _size * sizeof(T));
  if (error != cudaSuccess)
    Logger::print_err("Cuda Malloc error!");
}

template <typename T>
MultiArray<T>::MultiArray(const Extent& extent)
    : MultiArray(extent.width(), extent.height(), extent.depth()) {}

template <typename T>
MultiArray<T>::MultiArray(const self_type& other)
    : _extent(other._extent), _size(other._size), _dim(other._dim) {
  // _data = new T[_size];
  // _data = reinterpret_cast<T*>(aligned_malloc(_size * sizeof(T), 64));
  auto error = cudaMallocManaged(&_data, _size * sizeof(T));
  if (error != cudaSuccess)
    Logger::print_err("Cuda Malloc error!");
  copyFrom(other);
}

template <typename T>
MultiArray<T>::MultiArray(self_type&& other)
    : _extent(other._extent), _size(other._size), _dim(other._dim) {
  _data = other._data;
  other._data = nullptr;
  other._size = 0;
  other._extent.width() = 0;
  other._extent.height() = 1;
  other._extent.depth() = 1;
}

template <typename T>
MultiArray<T>::~MultiArray() {
  if (_data != nullptr) {
    // delete[] _data;
    cudaFree(_data);
    _data = nullptr;
  }
}

template <typename T>
auto
MultiArray<T>::operator=(const self_type& other) -> self_type& {
  if (_extent != other._extent) {
    resize(other._extent);
  }
  copyFrom(other);
  return (*this);
}

template <typename T>
auto
MultiArray<T>::operator=(self_type&& other) -> self_type& {
  if (_extent != other._extent) {
    _extent = other._extent;
    _size = _extent.size();
    find_dim();
  }
  // If the memory is already allocated, then pointing _data to
  // another place will lead to memory leak.
  if (_data != nullptr) {
    // delete[] _data;
    cudaFree(_data);
  }
  _data = other._data;
  other._data = nullptr;
  return (*this);
}

template <typename T>
void
MultiArray<T>::resize(int width, int height, int depth) {
  _extent.width() = width;
  _extent.height() = height;
  _extent.depth() = depth;
  _size = _extent.size();
  find_dim();
  if (_data != nullptr) {
    // delete[] _data;
    cudaFree(_data);
  }
  // _data = new T[_size];
  auto error = cudaMallocManaged(&_data, _size * sizeof(T));
  if (error != cudaSuccess)
    Logger::print_err("Cuda Malloc error!");
  assign( static_cast<T>(0) );
}

template <typename T>
void
MultiArray<T>::resize(Extent extent) {
  resize(extent.width(), extent.height(), extent.depth());
}

template <typename T>
void
MultiArray<T>::find_dim() {
  if (_extent.height() <= 1 && _extent.depth() <= 1)
    _dim = 1;
  else if (_extent.depth() <= 1)
    _dim = 2;
  else
    _dim = 3;
}

}

#endif   // _MULTI_ARRAY_IMPL_H_
