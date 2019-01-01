#ifndef _PTC_UPDATER_AVX_HELPER_H_
#define _PTC_UPDATER_AVX_HELPER_H_

#include "utils/simd.h"

namespace Aperture {

template <typename VF>
inline VF
center2d(const VF& sx0, const VF& sx1, const VF& sy0, const VF& sy1) {
  return mul_add(
             sx1, sy1 * 2.0,
             mul_add(sx0, sy1, mul_add(sx1, sy0, sx0 * sy0 * 2.0))) *
         0.1666667;
}

template <typename VF>
inline VF
movement3d(const VF& sx0, const VF& sx1, const VF& sy0, const VF& sy1,
           const VF& sz0, const VF& sz1) {
  return (sz1 - sz0) * center2d(sx0, sx1, sy0, sy1);
}

}  // namespace Aperture

#endif  // _PTC_UPDATER_AVX_HELPER_H_
