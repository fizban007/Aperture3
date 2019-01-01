#ifndef _SIMD_H_
#define _SIMD_H_

#include <immintrin.h>
#define MAX_VECTOR_SIZE 512
#include "vectorclass.h"

namespace Aperture {

namespace simd {

typedef __m256 v8f;
typedef __m256i v8i;
typedef __m256d v4d;

template <typename VF>
inline VF
lerp(const VF& x, const VF& a, const VF& b) {
  VF r = b - a;
  r *= x;
  return r + a;
}

template <typename T>
struct to_signed {
  typedef int type;
};

template <>
struct to_signed<Vec8ui> {
  typedef Vec8i type;
};

template <>
struct to_signed<Vec4uq> {
  typedef Vec4q type;
};

template <>
struct to_signed<Vec16ui> {
  typedef Vec16i type;
};

template <>
struct to_signed<Vec8uq> {
  typedef Vec8q type;
};

}  // namespace simd

}  // namespace Aperture

#endif  // _SIMD_H_
