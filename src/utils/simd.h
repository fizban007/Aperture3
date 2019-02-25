#ifndef _SIMD_H_
#define _SIMD_H_

#include <immintrin.h>
#include "core/typedefs.h"
#define MAX_VECTOR_SIZE 512
#include "vectorclass.h"

namespace Aperture {

namespace simd {

#if !defined(USE_DOUBLE) && \
    (defined(__AVX512F__) || defined(__AVX512__))
typedef Vec16ui Vec_ui_type;
typedef Vec16i Vec_i_type;
typedef Vec16ib Vec_ib_type;
typedef Vec16f Vec_f_type;
constexpr int vec_width = 16;
#elif defined(USE_DOUBLE) && \
    (defined(__AVX512F__) || defined(__AVX512__))
typedef Vec8uq Vec_ui_type;
typedef Vec8q Vec_i_type;
typedef Vec8qb Vec_ib_type;
typedef Vec8d Vec_f_type;
constexpr int vec_width = 8;
#elif !defined(USE_DOUBLE) && defined(__AVX2__)
typedef Vec8ui Vec_idx_type;
typedef Vec8ui Vec_ui_type;
typedef Vec8i Vec_i_type;
typedef Vec8ib Vec_ib_type;
typedef Vec8f Vec_f_type;
constexpr int vec_width = 8;
#elif defined(USE_DOUBLE) && defined(__AVX2__)
typedef Vec8ui Vec_idx_type;
typedef Vec4uq Vec_ui_type;
typedef Vec4q Vec_i_type;
typedef Vec4qb Vec_ib_type;
typedef Vec4d Vec_f_type;
constexpr int vec_width = 4;
#else
typedef uint32_t Vec_idx_type;
typedef uint32_t Vec_ui_type;
typedef int Vec_i_type;
typedef bool Vec_ib_type;
typedef float Vec_f_type;
constexpr int vec_width = 1;
#endif

template <typename VF>
inline VF
lerp(const VF& x, const VF& a, const VF& b) {
  VF r = b - a;
  r *= x;
  return r + a;
}

struct simd_buffer {
  Scalar x[vec_width] = {};

  simd_buffer() {}
  
  simd_buffer(Scalar v) {
    for (int i = 0; i < vec_width; i++)
      x[i] = v;
  }

  simd_buffer operator*(Scalar v) const {
    simd_buffer buf;
    for (int i = 0; i < vec_width; i++)
      buf.x[i] *= v;
    return buf;
  }

  simd_buffer operator+(const simd_buffer& buf) const {
    simd_buffer result;
    for (int i = 0; i < vec_width; i++)
      result.x[i] = buf.x[i] + x[i];
    return result;
  }

  simd_buffer& operator=(Scalar f) {
    for (int i = 0; i < vec_width; i++)
      x[i] = f;
    return *this;
  }
};

inline simd_buffer
operator*(Scalar v, const simd_buffer& buffer) {
  return buffer * v;
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
