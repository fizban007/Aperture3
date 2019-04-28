#ifndef _TYPED_PITCHEDPTR_CUH_
#define _TYPED_PITCHEDPTR_CUH_

#include "cuda/cuda_control.h"
#include <cuda_runtime.h>

namespace Aperture {

template <typename T>
struct typed_pitchedptr {
  typedef typed_pitchedptr<T> self_type;
  cudaPitchedPtr p;

  HOST_DEVICE typed_pitchedptr() {}
  HOST_DEVICE typed_pitchedptr(cudaPitchedPtr ptr) : p(ptr) {}
  HOST_DEVICE typed_pitchedptr(const self_type& other) : p(other.p) {}

  HOST_DEVICE operator cudaPitchedPtr() const { return p; }

  HD_INLINE self_type& operator=(const self_type& other) {
    p = other.p;
    return *this;
  }

  HD_INLINE T& operator[](size_t offset) {
    return *(T*)((char*)p.ptr + offset);
  }

  HD_INLINE const T& operator[](size_t offset) const {
    return *(T*)((char*)p.ptr + offset);
  }

  HD_INLINE T& operator()(int i, int j = 0) {
    return *((T*)((char*)p.ptr + j * p.pitch) + i);
  }

  HD_INLINE const T& operator()(int i, int j = 0) const {
    return *((T*)((char*)p.ptr + j * p.pitch) + i);
  }

  HD_INLINE T& operator()(int i, int j, int k) {
    return *((T*)((char*)p.ptr + (j + k * p.ysize) * p.pitch) + i);
  }

  HD_INLINE const T& operator()(int i, int j, int k) const {
    return *((T*)((char*)p.ptr + (j + k * p.ysize) * p.pitch) + i);
  }

  HD_INLINE size_t compute_offset(int i, int j = 0) const {
    return j * p.pitch + i * sizeof(T);
  }

  HD_INLINE size_t compute_offset(int i, int j, int k) const {
    return (j + k * p.ysize) * p.pitch + i * sizeof(T);
  }
};

}  // namespace Aperture

#endif  // _TYPED_PITCHEDPTR_CUH_