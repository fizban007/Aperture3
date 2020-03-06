#ifndef _PTC_UPDATER_HELPER_H_
#define _PTC_UPDATER_HELPER_H_

#include "core/typedefs.h"
#include "cuda/utils/interpolation.cuh"

namespace Aperture {

namespace Kernels {

typedef Spline::spline_t<1> spline_t;
// typedef Spline::spline_t<3> spline_t;

// HD_INLINE Scalar
// cloud_in_cell_f(Scalar dx) {
//   return max(1.0 - std::abs(dx), 0.0);
// }

// HD_INLINE Scalar
// interpolate(Scalar rel_pos, int ptc_cell, int target_cell) {
//   Scalar dx =
//       ((Scalar)target_cell + 0.5 - (rel_pos + Scalar(ptc_cell)));
//   return cloud_in_cell_f(dx);
// }

HD_INLINE Scalar
center2d(Scalar sx0, Scalar sx1, Scalar sy0, Scalar sy1) {
  return (2.0f * sx1 * sy1 + sx0 * sy1 + sx1 * sy0 + 2.0f * sx0 * sy0) *
         0.1666667f;
}

HD_INLINE Scalar
movement3d(Scalar sx0, Scalar sx1, Scalar sy0, Scalar sy1, Scalar sz0,
           Scalar sz1) {
  return (sz1 - sz0) * center2d(sx0, sx1, sy0, sy1);
}

HD_INLINE Scalar
movement2d(Scalar sx0, Scalar sx1, Scalar sy0, Scalar sy1) {
  return (sy1 - sy0) * 0.5f * (sx0 + sx1);
}

// union float2UllUnion {
//   // double d;
//   float2 f;
//   unsigned long long int ull;
// };

// __device__ __forceinline__ void
// atomicAddKbn(float2* __restrict address, const float val) {
//   unsigned long long int* address_as_ull =
//       (unsigned long long int*)address;
//   float2UllUnion old, assumed, tmp;
//   old.ull = *address_as_ull;
//   do {
//     assumed = old;
//     tmp = assumed;

//     // Kahan and Babuska summation, Neumaier variant
//     float t = tmp.f.x + val;
//     tmp.f.y += (fabsf(tmp.f.x) >= fabsf(val)) ? ((tmp.f.x - t) + val)
//                                               : ((val - t) + tmp.f.x);
//     tmp.f.x = t;

//     old.ull = atomicCAS(address_as_ull, assumed.ull, tmp.ull);

//   } while (assumed.ull != old.ull);
// }

// __device__ __forceinline__ void
// atomicAddKahan(float2* __restrict address, const float val) {
//   unsigned long long int* address_as_ull =
//       (unsigned long long int*)address;
//   float2UllUnion old, assumed, tmp;
//   old.ull = *address_as_ull;
//   do {
//     assumed = old;
//     tmp = assumed;
//     // kahan summation
//     const float y = val - tmp.f.y;
//     const float t = tmp.f.x + y;
//     tmp.f.y = (t - tmp.f.x) - y;
//     tmp.f.x = t;

//     old.ull = atomicCAS(address_as_ull, assumed.ull, tmp.ull);

//   } while (assumed.ull != old.ull);
// }

}  // namespace Kernels

}  // namespace Aperture

#endif
