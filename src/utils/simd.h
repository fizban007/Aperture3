#ifndef _SIMD_H_
#define _SIMD_H_

#include <immintrin.h>
#include "vectorclass.h"

namespace Aperture {

namespace simd {

#if defined(__AVX2__)

typedef __m256 v8f;
typedef __m256i v8i;
typedef __m256d v4d;

inline Vec8f lerp(const Vec8f& x, const Vec8f& a, const Vec8f& b) {
  Vec8f r = b - a;
  r *= x;
  return r + a;
}

#endif

}


}

#endif  // _SIMD_H_
