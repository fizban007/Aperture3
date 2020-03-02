#ifndef _PITCHPTR_CUH_
#define _PITCHPTR_CUH_

#include "core/vec3.h"
#include "core/multi_array.h"
#include "core/fields.h"
#include "cuda/cuda_control.h"
// #include <cuda_runtime.h>

namespace Aperture {

/// A struct that wraps the cudaPitchedptr and provides some simple
/// access functionality
template <typename T>
struct pitchptr {
  typedef pitchptr<T> self_type;
  cudaPitchedPtr p;

  HOST_DEVICE pitchptr() {}
  HOST_DEVICE pitchptr(cudaPitchedPtr ptr) : p(ptr) {}
  HOST_DEVICE pitchptr(const self_type& other) : p(other.p) {}

  // HOST_DEVICE operator cudaPitchedPtr() const { return p; }

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

  HD_INLINE T& operator()(const Index& idx) {
    return operator()(idx.x, idx.y, idx.z);
  }

  HD_INLINE const T& operator()(const Index& idx) const {
    return operator()(idx.x, idx.y, idx.z);
  }

  HD_INLINE size_t compute_offset(int i, int j = 0) const {
    return j * p.pitch + i * sizeof(T);
  }

  HD_INLINE size_t compute_offset(int i, int j, int k) const {
    return (j + k * p.ysize) * p.pitch + i * sizeof(T);
  }
};

template <typename T>
pitchptr<T> get_pitchptr(multi_array<T>& array) {
  return pitchptr<T>(get_cudaPitchedPtr(array));
}

template <int N, typename T>
pitchptr<T> get_pitchptr(field<N, T>& field, int n = 0) {
  return pitchptr<T>(get_cudaPitchedPtr(field.data(n)));
}

template <typename T>
cudaPitchedPtr get_cudaPitchedPtr(multi_array<T>& array) {
  return make_cudaPitchedPtr(array.dev_ptr(), array.pitch(),
                             array.extent().width(),
                             array.extent().height());
}

template <int N, typename T>
cudaPitchedPtr get_cudaPitchedPtr(field<N, T>& field, int n) {
  return get_cudaPitchedPtr(field.data(n));
}

}  // namespace Aperture

#endif  // _PITCHPTR_CUH_

// Local Variables:
// mode: cuda
// End:
