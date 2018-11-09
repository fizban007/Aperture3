#ifndef _CUDARNG_H_
#define _CUDARNG_H_

#include "cuda/cuda_control.h"
#include <curand_kernel.h>

namespace Aperture {

namespace Kernels {

// Helper struct to plug into inverse compton module
struct CudaRng {
  HOST_DEVICE CudaRng(curandState* state) : m_state(state) {
    m_local_state = *state;
  }
  HOST_DEVICE ~CudaRng() {
    *m_state = m_local_state;
  }

  // Generates a device random number between 0.0 and 1.0
  __device__ __forceinline__ float operator()() {
    return curand_uniform(&m_local_state);
  }

  curandState* m_state;
  curandState m_local_state;
};

}  // namespace Kernels

}  // namespace Aperture

#endif  // _CUDARNG_H_
