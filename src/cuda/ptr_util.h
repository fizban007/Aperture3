#ifndef _CUDA_PTR_UTIL_H_
#define _CUDA_PTR_UTIL_H_

#include "cuda_runtime.h"

namespace Aperture {

HD_INLINE Scalar*
ptrAddr(cudaPitchedPtr p, size_t offset) {
  return (Scalar*)((char*)p.ptr + offset);
}

HD_INLINE Scalar*
ptrAddr(cudaPitchedPtr p, int i, int j) {
  return (Scalar*)((char*)p.ptr + j * p.pitch + i * sizeof(Scalar));
}

HD_INLINE double*
ptrAddr_d(cudaPitchedPtr p, size_t offset) {
  return (double*)((char*)p.ptr + offset);
}

} // namespace Aperture


#endif  // _CUDA_PTR_UTIL_H_
