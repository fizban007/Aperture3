#ifndef _GRAVITY_CUH_
#define _GRAVITY_CUH_

#include "cuda/constant_mem.h"

namespace Aperture {

namespace Kernels {

__device__ Scalar alpha_gr(Scalar r);

__device__ __forceinline__ void
gravity(Scalar& p1, Scalar p2, Scalar p3, Scalar& gamma, Scalar r,
        int sp, Scalar dt) {
  // Add an artificial gravity
  p1 -= dt * dev_params.gravity * dev_masses[sp] / (r * r * std::abs(dev_charges[sp]));
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

}  // namespace Kernels
}  // namespace Aperture

#endif
