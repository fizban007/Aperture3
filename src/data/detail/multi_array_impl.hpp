#ifndef _MULTI_ARRAY_IMPL_H_
#define _MULTI_ARRAY_IMPL_H_

#include "cuda/cudaUtility.h"
#include "cuda_runtime.h"
#include "data/detail/multi_array_utils.hpp"
#include "data/multi_array.h"
#include "thrust/copy.h"
#include "thrust/device_ptr.h"
#include "utils/logger.h"
#include "utils/memory.h"

namespace Aperture {

template <typename T>
inline cudaExtent
cuda_ext(const Extent& ext, const T& t) {
  return make_cudaExtent(ext.x * sizeof(T), ext.y, ext.z);
}

template <typename T>
MultiArray<T>::MultiArray()
    : _data_d(make_cudaPitchedPtr(nullptr, 0, 0, 0)),
      _data_h(nullptr),
      _size(0) {
  _extent.width() = 0;
  _extent.height() = 1;
  _extent.depth() = 1;
  find_dim();
}

template <typename T>
MultiArray<T>::MultiArray(int width, int height, int depth,
                          int deviceId)
    : _data_d(make_cudaPitchedPtr(nullptr, 0, 0, 0)),
      _data_h(nullptr),
      _extent{width, height, depth} {
  _size = _extent.size();
  // Logger::print_info("extent has {}, {}, {}", _extent.width(),
  //                    _extent.height(), _extent.depth());
  // auto ext = cuda_ext(_extent, T{});
  // Logger::print_info("now cuda extent has {}, {}, {}", ext.width,
  //                    ext.height, ext.depth);
  find_dim();
  alloc_mem(_extent, deviceId);
}

template <typename T>
MultiArray<T>::MultiArray(const Extent& extent, int deviceId)
    : MultiArray(extent.width(), extent.height(), extent.depth(),
                 deviceId) {}

template <typename T>
MultiArray<T>::MultiArray(const self_type& other)
    : _data_d(make_cudaPitchedPtr(nullptr, 0, 0, 0)),
      _data_h(nullptr),
      _extent(other._extent),
      _size(other._size),
      _dim(other._dim) {
  // _data = new T[_size];
  // _data = reinterpret_cast<T*>(aligned_malloc(_size * sizeof(T),
  // 64));
  alloc_mem(_extent, other._devId);
  copyFrom(other);
}

template <typename T>
MultiArray<T>::MultiArray(self_type&& other)
    : _extent(other._extent), _size(other._size), _dim(other._dim) {
  _data_d = other._data_d;
  _data_h = other._data_h;
  other._data_d.ptr = nullptr;
  other._data_h = nullptr;
  other._size = 0;
  other._extent.width() = 0;
  other._extent.height() = 1;
  other._extent.depth() = 1;
}

template <typename T>
MultiArray<T>::~MultiArray() {
  free_mem();
}

template <typename T>
void
MultiArray<T>::alloc_mem(const Extent& ext, int deviceId) {
  if (_data_d.ptr != nullptr || _data_h != nullptr) free_mem();
  _devId = deviceId;
  CudaSafeCall(cudaSetDevice(_devId));
  auto extent = cuda_ext(ext, T{});
  // Logger::print_info("extent has {}, {}, {}", extent.width,
  //                    extent.height, extent.depth);
  CudaSafeCall(cudaMalloc3D(&_data_d, extent));
  // Logger::print_info("pitch is {}, xsize is {}, ysize is {}",
  //                    _data_d.pitch, _data_d.xsize, _data_d.ysize);
  _data_h = new T[ext.size()];
}

template <typename T>
void
MultiArray<T>::free_mem() {
  CudaSafeCall(cudaSetDevice(_devId));
  if (_data_d.ptr != nullptr) {
    CudaSafeCall(cudaFree(_data_d.ptr));
    _data_d.ptr = nullptr;
  }
  if (_data_h != nullptr) {
    delete[] _data_h;
    _data_h = nullptr;
  }
}

template <typename T>
void
MultiArray<T>::copyFrom(const self_type& other) {
  assert(_size == other._size);
  CudaSafeCall(cudaSetDevice(_devId));
  cudaMemcpy3DParms myParms = {0};
  myParms.srcPtr = other._data_d;
  myParms.srcPos = make_cudaPos(0, 0, 0);
  myParms.dstPtr = _data_d;
  myParms.dstPos = make_cudaPos(0, 0, 0);
  myParms.extent = cuda_ext(_extent, T{});
  myParms.kind = cudaMemcpyDeviceToDevice;

  CudaSafeCall(cudaMemcpy3D(&myParms));
  sync_to_host();
  // auto ptr = thrust::device_pointer_cast(_data_d);
  // auto ptr_other = thrust::device_pointer_cast(other._data_d);
  // thrust::copy_n(ptr_other, _size, ptr);
}

template <typename T>
void
MultiArray<T>::assign(const data_type& value) {
  std::fill_n(_data_h, _size, value);
}

template <typename T>
void
MultiArray<T>::assign_dev(const data_type& value) {
  CudaSafeCall(cudaSetDevice(_devId));
  if (_dim == 3) {
    // Logger::print_info("assign_dev 3d version");
    dim3 blockSize(8, 8, 8);
    // dim3 gridSize(8, 8, 8);
    dim3 gridSize((_extent.x + 7)/8,
                  (_extent.y + 7)/8,
                  (_extent.z + 7)/8);
    Kernels::map_array_unary_op<T><<<gridSize, blockSize>>>(
        _data_d, _extent, detail::Op_AssignConst<T>(value));
    CudaCheckError();
  } else if (_dim == 2) {
    // Logger::print_info("assign_dev 2d version");
    dim3 blockSize(32, 16);
    dim3 gridSize((_extent.x+31)/32, (_extent.y+15)/16);
    Kernels::map_array_unary_op_2d<T><<<gridSize, blockSize>>>(
        _data_d, _extent, detail::Op_AssignConst<T>(value));
    CudaCheckError();
  }
}

template <typename T>
auto
MultiArray<T>::operator=(const self_type& other) -> self_type& {
  if (_extent != other._extent) {
    resize(other._extent, other._devId);
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
  free_mem();
  _data_d = other._data_d;
  other._data_d.ptr = nullptr;
  _data_h = other._data_h;
  other._data_h = nullptr;
  _devId = other._devId;
  return (*this);
}

template <typename T>
void
MultiArray<T>::resize(int width, int height, int depth, int deviceId) {
  _extent.width() = width;
  _extent.height() = height;
  _extent.depth() = depth;
  _size = _extent.size();
  find_dim();
  free_mem();
  alloc_mem(_extent, deviceId);
}

template <typename T>
void
MultiArray<T>::resize(Extent extent, int deviceId) {
  resize(extent.width(), extent.height(), extent.depth(), deviceId);
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

template <typename T>
void
MultiArray<T>::sync_to_device(int devId) {
  CudaSafeCall(cudaSetDevice(devId));
  // CudaSafeCall(cudaMemPrefetchAsync(_data, _size * sizeof(T),
  // devId));
  cudaMemcpy3DParms myParms = {0};
  myParms.srcPtr = make_cudaPitchedPtr(
      (void*)_data_h, _extent.x * sizeof(T), _extent.x, _extent.y);
  myParms.srcPos = make_cudaPos(0, 0, 0);
  myParms.dstPtr = _data_d;
  myParms.dstPos = make_cudaPos(0, 0, 0);
  myParms.extent = cuda_ext(_extent, T{});
  myParms.kind = cudaMemcpyHostToDevice;
  // Logger::print_info("before copy to device, extent has {}, {}, {}",
  // myParms.extent.width,
  //                    myParms.extent.height, myParms.extent.depth);

  CudaSafeCall(cudaMemcpy3D(&myParms));
}

template <typename T>
void
MultiArray<T>::sync_to_device() {
  sync_to_device(_devId);
}

template <typename T>
void
MultiArray<T>::sync_to_host() {
  CudaSafeCall(cudaSetDevice(_devId));
  // CudaSafeCall(cudaMemPrefetchAsync(_data, _size * sizeof(T),
  // cudaCpuDeviceId));
  cudaMemcpy3DParms myParms = {0};
  myParms.srcPtr = _data_d;
  myParms.srcPos = make_cudaPos(0, 0, 0);
  myParms.dstPtr = make_cudaPitchedPtr(
      (void*)_data_h, _extent.x * sizeof(T), _extent.x, _extent.y);
  myParms.dstPos = make_cudaPos(0, 0, 0);
  myParms.extent = cuda_ext(_extent, T{});
  myParms.kind = cudaMemcpyDeviceToHost;
  // Logger::print_info("before copy to host, extent has {}, {}, {}",
  // myParms.extent.width,
  //                    myParms.extent.height, myParms.extent.depth);
  // Logger::print_info("host pitchedptr has has {}, {}, {}",
  // myParms.dstPtr.pitch,
  //                    myParms.dstPtr.xsize, myParms.dstPtr.ysize);

  CudaSafeCall(cudaMemcpy3D(&myParms));
}

}  // namespace Aperture

#endif  // _MULTI_ARRAY_IMPL_H_
