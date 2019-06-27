#ifndef _GRAVITY_CUH_
#define _GRAVITY_CUH_

#include "cuda/constant_mem.h"

namespace Aperture {

namespace Kernels {

__device__ Scalar alpha_gr(Scalar r);

__device__ __forceinline__ void
gravity(Scalar& p1, Scalar& p2, Scalar& p3, Scalar& gamma, Scalar& r,
        Scalar& dt) {
  // Add an artificial gravity
  if (dev_params.gravity_on) {
    p1 -= dt * alpha_gr(r) * dev_params.gravity / (r * r * r);
    gamma = sqrt(1.0f + p1 * p1 + p2 * p2 + p3 * p3);
    if (gamma != gamma) {
      printf(
          "NaN detected after gravity! p1 is %f, p2 is %f, p3 is "
          "%f, gamma is "
          "%f\n",
          p1, p2, p3, gamma);
      asm("trap;");
    }
  }
}

}  // namespace Kernels
}  // namespace Aperture

#endif
