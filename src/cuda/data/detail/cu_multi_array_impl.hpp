#ifndef _CU_MULTI_ARRAY_IMPL_H_
#define _CU_MULTI_ARRAY_IMPL_H_

// #include "core/detail/multi_array_impl.hpp"
#include "core/detail/multi_array_utils.hpp"
#include "cuda/cudaUtility.h"
#include "cuda/data/cu_multi_array.h"
#include "utils/logger.h"
#include <thrust/copy.h>
#include <thrust/device_ptr.h>

namespace Aperture {

template <typename T>
inline cudaExtent
cuda_ext(const Extent& ext, const T& t) {
  return make_cudaExtent(ext.x * sizeof(T), ext.y, ext.z);
}

template <typename T>
cu_multi_array<T>::cu_multi_array()
    : multi_array<T>(),
      _data_d(make_cudaPitchedPtr(nullptr, 0, 0, 0)) {}

template <typename T>
cu_multi_array<T>::cu_multi_array(int width, int height, int depth,
                                  int deviceId)
    : multi_array<T>(width, height, depth),
      _data_d(make_cudaPitchedPtr(nullptr, 0, 0, 0)) {
  // Logger::print_info("extent has {}, {}, {}", _extent.width(),
  //                    _extent.height(), _extent.depth());
  // auto ext = cuda_ext(_extent, T{});
  // Logger::print_info("now cuda extent has {}, {}, {}", ext.width,
  //                    ext.height, ext.depth);
  alloc_mem(this->_extent, deviceId);
}

template <typename T>
cu_multi_array<T>::cu_multi_array(const Extent& extent, int deviceId)
    : cu_multi_array(extent.width(), extent.height(), extent.depth(),
                     deviceId) {}

template <typename T>
cu_multi_array<T>::cu_multi_array(const self_type& other)
    : multi_array<T>(other),
      _data_d(make_cudaPitchedPtr(nullptr, 0, 0, 0)),
      _devId(other._devId) {
  // _data = new T[_size];
  // _data = reinterpret_cast<T*>(aligned_malloc(_size * sizeof(T),
  // 64));
  alloc_mem(this->_extent, other._devId);
  copy_from(other);
}

template <typename T>
cu_multi_array<T>::cu_multi_array(self_type&& other)
    : multi_array<T>(other) {
  _data_d = other._data_d;
  _devId = other._devId;
  other._data_d.ptr = nullptr;
}

template <typename T>
cu_multi_array<T>::~cu_multi_array() {
  free_mem();
}

template <typename T>
void
cu_multi_array<T>::alloc_mem(const Extent& ext, int deviceId) {
  if (_data_d.ptr != nullptr) free_mem();
  _devId = deviceId;
  CudaSafeCall(cudaSetDevice(_devId));
  auto extent = cuda_ext(ext, T{});
  // Logger::print_info("extent has {}, {}, {}", extent.width,
  //                    extent.height, extent.depth);
  CudaSafeCall(cudaMalloc3D(&_data_d, extent));
  // Logger::print_info("pitch is {}, xsize is {}, ysize is {}",
  //                    _data_d.pitch, _data_d.xsize, _data_d.ysize);
  // base_class::alloc_mem(ext);
}

template <typename T>
void
cu_multi_array<T>::free_mem() {
  CudaSafeCall(cudaSetDevice(_devId));
  if (_data_d.ptr != nullptr) {
    CudaSafeCall(cudaFree(_data_d.ptr));
    _data_d.ptr = nullptr;
  }
}

template <typename T>
void
cu_multi_array<T>::copy_from(const self_type& other) {
  assert(this->_size == other._size);
  CudaSafeCall(cudaSetDevice(_devId));
  cudaMemcpy3DParms myParms = {0};
  myParms.srcPtr = other._data_d;
  myParms.srcPos = make_cudaPos(0, 0, 0);
  myParms.dstPtr = _data_d;
  myParms.dstPos = make_cudaPos(0, 0, 0);
  myParms.extent = cuda_ext(this->_extent, T{});
  myParms.kind = cudaMemcpyDeviceToDevice;

  CudaSafeCall(cudaMemcpy3D(&myParms));
  sync_to_host();
  // auto ptr = thrust::device_pointer_cast(_data_d);
  // auto ptr_other = thrust::device_pointer_cast(other._data_d);
  // thrust::copy_n(ptr_other, _size, ptr);
}

template <typename T>
void
cu_multi_array<T>::assign(const data_type& value) {
  // std::fill_n(_data_h, _size, value);
  base_class::assign(value);
}

template <typename T>
void
cu_multi_array<T>::assign_dev(const data_type& value) {
  CudaSafeCall(cudaSetDevice(_devId));
  if (this->_dim == 3) {
    // Logger::print_info("assign_dev 3d version");
    dim3 blockSize(8, 8, 8);
    // dim3 gridSize(8, 8, 8);
    dim3 gridSize((this->_extent.x + 7) / 8, (this->_extent.y + 7) / 8,
                  (this->_extent.z + 7) / 8);
    Kernels::map_array_unary_op<T><<<gridSize, blockSize>>>(
        _data_d, this->_extent, detail::Op_AssignConst<T>(value));
    CudaCheckError();
  } else if (this->_dim == 2) {
    // Logger::print_info("assign_dev 2d version");
    dim3 blockSize(32, 16);
    dim3 gridSize((this->_extent.x + 31) / 32,
                  (this->_extent.y + 15) / 16);
    Kernels::map_array_unary_op_2d<T><<<gridSize, blockSize>>>(
        _data_d, this->_extent, detail::Op_AssignConst<T>(value));
    CudaCheckError();
  }
}

template <typename T>
auto
cu_multi_array<T>::operator=(const self_type& other) -> self_type& {
  if (this->_extent != other._extent) {
    resize(other._extent, other._devId);
  }
  copy_from(other);
  return (*this);
}

template <typename T>
auto
cu_multi_array<T>::operator=(self_type&& other) -> self_type& {
  if (this->_extent != other._extent) {
    this->_extent = other._extent;
    this->_size = this->_extent.size();
    this->find_dim();
  }
  // If the memory is already allocated, then pointing _data to
  // another place will lead to memory leak.
  free_mem();
  base_class::free_mem();
  _data_d = other._data_d;
  other._data_d.ptr = nullptr;
  this->_data = other._data;
  other._data = nullptr;
  _devId = other._devId;
  return (*this);
}

template <typename T>
void
cu_multi_array<T>::resize(int width, int height, int depth,
                          int deviceId) {
  // this->_extent.width() = width;
  // this->_extent.height() = height;
  // this->_extent.depth() = depth;
  // this->_size = this->_extent.size();
  // this->find_dim();
  // base_class::free_mem();
  base_class::resize(width, height, depth);
  free_mem();
  alloc_mem(this->_extent, deviceId);
}

template <typename T>
void
cu_multi_array<T>::resize(Extent extent, int deviceId) {
  resize(extent.width(), extent.height(), extent.depth(), deviceId);
}

template <typename T>
void
cu_multi_array<T>::sync_to_device(int devId) {
  CudaSafeCall(cudaSetDevice(devId));
  // CudaSafeCall(cudaMemPrefetchAsync(_data, _size * sizeof(T),
  // devId));
  cudaMemcpy3DParms myParms = {0};
  myParms.srcPtr =
      make_cudaPitchedPtr(this->_data, this->_pitch,
                          this->_extent.x, this->_extent.y);
  myParms.srcPos = make_cudaPos(0, 0, 0);
  myParms.dstPtr = _data_d;
  myParms.dstPos = make_cudaPos(0, 0, 0);
  myParms.extent = cuda_ext(this->_extent, T{});
  myParms.kind = cudaMemcpyHostToDevice;
  // Logger::print_info("before copy to device, extent has {}, {}, {}",
  // myParms.extent.width,
  //                    myParms.extent.height, myParms.extent.depth);

  CudaSafeCall(cudaMemcpy3D(&myParms));
}

template <typename T>
void
cu_multi_array<T>::sync_to_device() {
  sync_to_device(_devId);
}

template <typename T>
void
cu_multi_array<T>::sync_to_host() {
  CudaSafeCall(cudaSetDevice(_devId));
  // CudaSafeCall(cudaMemPrefetchAsync(_data, _size * sizeof(T),
  // cudaCpuDeviceId));
  cudaMemcpy3DParms myParms = {0};
  myParms.srcPtr = _data_d;
  myParms.srcPos = make_cudaPos(0, 0, 0);
  myParms.dstPtr =
      make_cudaPitchedPtr(this->_data, this->_pitch,
                          this->_extent.x, this->_extent.y);
  myParms.dstPos = make_cudaPos(0, 0, 0);
  myParms.extent = cuda_ext(this->_extent, T{});
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

#endif  // _CU_MULTI_ARRAY_IMPL_H_
