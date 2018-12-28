#ifndef _INTERPOLATION_H_
#define _INTERPOLATION_H_

#include <cmath>
#include <utility>
// #include "cuda/cuda_control.h"
// #include "data/fields_dev.h"
#include "data/multi_array.h"
#include "data/stagger.h"
#include "data/typedefs.h"
#include "data/vec3.h"
#include "vectorclass.h"
// #include "boost/fusion/container/vector.hpp"

#ifndef __CUDACC__
using std::abs;
using std::max;
#endif

namespace Aperture {

namespace detail {

struct interp_nearest_grid_point {
  enum { radius = 1, support = 1 };

  double operator()(double dx) const {
    return (std::abs(dx) <= 0.5 ? 1.0 : 0.0);
  }
};

struct interp_cloud_in_cell {
  enum { radius = 1, support = 2 };

  double operator()(double dx) const {
    double abs_dx = std::abs(dx);
    // return (abs_dx < 0.5 ? max(1.0 - abs_dx, 0.0) : abs_dx);
    return max(1.0 - abs_dx, 0.0);
  }
};

struct interp_triangular_shaped_cloud {
  enum { radius = 2, support = 3 };

  double operator()(double dx) const {
    double abs_dx = std::abs(dx);
    if (abs_dx < 0.5) {
      return 0.75 - dx * dx;
    } else if (abs_dx < 1.5) {
      double tmp = 1.5 - abs_dx;
      return 0.5 * tmp * tmp;
    } else {
      return 0.0;
    }
  }
};

struct interp_piecewise_cubic {
  enum { radius = 2, support = 4 };

  double operator()(double dx) const {
    double abs_dx = std::abs(dx);
    if (abs_dx < 1.0) {
      double tmp = abs_dx * abs_dx;
      return 2.0 / 3.0 - tmp + 0.5 * tmp * abs_dx;
    } else if (abs_dx < 2.0) {
      double tmp = 2.0 - abs_dx;
      return 1.0 / 6.0 * tmp * tmp * tmp;
    } else {
      return 0.0;
    }
  }
};
}  // namespace detail

// template <int Order>
class Interpolator {
 public:
  Interpolator(int order = 1) : m_order(order) {}
  ~Interpolator() {}

  template <typename FloatT>
  double interp_cell(FloatT pos, int p_cell, int target_cell,
                     int stagger = 0) const {
    FloatT x = ((double)target_cell + (stagger == 0 ? 0.5 : 1.0)) -
               (pos + (double)p_cell);
    switch (m_order) {
      case 0:
        return m_interp_0(x);
      case 1:
        return m_interp_1(x);
      case 2:
        return m_interp_2(x);
      case 3:
        return m_interp_3(x);
      default:
        return 0.0;
    }
  }

  template <typename FloatT>
  Vec3<Scalar> interp_cell(const Vec3<FloatT>& pos,
                           const Vec3<int>& p_cell,
                           const Vec3<int>& target_cell,
                           Stagger stagger = Stagger(0b000)) const {
    return Vec3<Scalar>(
        interp_cell(pos[0], p_cell[0], target_cell[0], stagger[0]),
        interp_cell(pos[1], p_cell[1], target_cell[1], stagger[1]),
        interp_cell(pos[2], p_cell[2], target_cell[2], stagger[2]));
  }

  template <typename FloatT>
  Scalar interp_weight(const Vec3<FloatT>& pos, const Vec3<int>& p_cell,
                       const Vec3<int>& target_cell,
                       Stagger stagger = Stagger(0b000)) const {
    return interp_cell(pos[0], p_cell[0], target_cell[0], stagger[0]) *
           interp_cell(pos[1], p_cell[1], target_cell[1], stagger[1]) *
           interp_cell(pos[2], p_cell[2], target_cell[2], stagger[2]);
  }

  int radius() const {
    switch (m_order) {
      case 0:
        return detail::interp_nearest_grid_point::radius;
      case 1:
        return detail::interp_cloud_in_cell::radius;
      case 2:
        return detail::interp_triangular_shaped_cloud::radius;
      case 3:
        return detail::interp_piecewise_cubic::radius;
      default:
        return 0.0;
    }
  }

  int support() const {
    switch (m_order) {
      case 0:
        return detail::interp_nearest_grid_point::support;
      case 1:
        return detail::interp_cloud_in_cell::support;
      case 2:
        return detail::interp_triangular_shaped_cloud::support;
      case 3:
        return detail::interp_piecewise_cubic::support;
      default:
        return 0.0;
    }
  }

 private:
  detail::interp_nearest_grid_point m_interp_0;
  detail::interp_cloud_in_cell m_interp_1;
  detail::interp_triangular_shaped_cloud m_interp_2;
  detail::interp_piecewise_cubic m_interp_3;
  int m_order;
};  // ----- end of class interpolator -----

#ifdef __AVX2__
inline Vec8f
interpolate(const multi_array<float>& data, const Vec8ui& cells,
            Vec8f x1, Vec8f x2, Vec8f x3, Stagger stagger) {
  Vec8ui d = cells / Divisor_ui(data.width());
  Vec8ui c1s = cells - d * data.width();
  Vec8ui offsets = c1s * sizeof(float) + d * data.pitch();
  uint32_t k_off = data.pitch() * data.height();

  Vec8i nx1 = select((bool)stagger[0], 0, truncate_to_int(x1 + 0.5));
  Vec8i nx2 = select((bool)stagger[1], 0, truncate_to_int(x2 + 0.5));
  Vec8i nx3 = select((bool)stagger[2], 0, truncate_to_int(x3 + 0.5));
  x1 = select((bool)stagger[0], x1, x1 + 0.5 - to_float(nx1));
  x2 = select((bool)stagger[1], x2, x2 + 0.5 - to_float(nx2));
  x3 = select((bool)stagger[2], x3, x3 + 0.5 - to_float(nx3));
  offsets += nx1 * sizeof(float);
  offsets += nx2 * data.pitch();
  offsets += nx3 * k_off;

  Vec8f f000 = _mm256_i32gather_ps(
      (float*)data.data(),
      offsets - (k_off + sizeof(float) + data.pitch()), 1);
  Vec8f f001 = _mm256_i32gather_ps((float*)data.data(),
                                   offsets - (k_off + data.pitch()), 1);
  Vec8f f010 = _mm256_i32gather_ps(
      (float*)data.data(), offsets - (sizeof(float) + k_off), 1);
  Vec8f f011 =
      _mm256_i32gather_ps((float*)data.data(), offsets - k_off, 1);
  Vec8f f100 = _mm256_i32gather_ps(
      (float*)data.data(), offsets - (sizeof(float) + data.pitch()), 1);
  Vec8f f101 = _mm256_i32gather_ps((float*)data.data(),
                                   offsets - data.pitch(), 1);
  Vec8f f110 = _mm256_i32gather_ps((float*)data.data(),
                                   offsets - sizeof(float), 1);
  Vec8f f111 = _mm256_i32gather_ps((float*)data.data(), offsets, 1);

  f000 = simd::lerp(x3, f000, f100);
  f010 = simd::lerp(x3, f010, f110);
  f001 = simd::lerp(x3, f001, f101);
  f011 = simd::lerp(x3, f011, f111);

  f000 = simd::lerp(x2, f000, f010);
  f001 = simd::lerp(x2, f001, f011);

  return simd::lerp(x1, f000, f001);
}
#endif

#ifdef __AVX512__
inline Vec16f
interpolate(const multi_array<float>& data, const Vec8ui& cells,
            Vec16f x1, Vec16f x2, Vec16f x3, Stagger stagger) {
  Vec16ui d = cells / Divisor_ui(data.width());
  Vec16ui c1s = cells - d * data.width();
  Vec16ui offsets = c1s * sizeof(float) + d * data.pitch();
  uint32_t k_off = data.pitch() * data.height();

  Vec16i nx1 = select((bool)stagger[0], 0, truncate_to_int(x1 + 0.5));
  Vec16i nx2 = select((bool)stagger[1], 0, truncate_to_int(x2 + 0.5));
  Vec16i nx3 = select((bool)stagger[2], 0, truncate_to_int(x3 + 0.5));
  x1 = select((bool)stagger[0], x1, x1 + 0.5 - to_float(nx1));
  x2 = select((bool)stagger[1], x2, x2 + 0.5 - to_float(nx2));
  x3 = select((bool)stagger[2], x3, x3 + 0.5 - to_float(nx3));
  offsets += nx1 * sizeof(float);
  offsets += nx2 * data.pitch();
  offsets += nx3 * k_off;

  Vec16f f000 = _mm512_i32gather_ps(
      (float*)data.data(),
      offsets - (k_off + sizeof(float) + data.pitch()), 1);
  Vec16f f001 = _mm512_i32gather_ps((float*)data.data(),
                                   offsets - (k_off + data.pitch()), 1);
  Vec16f f010 = _mm512_i32gather_ps(
      (float*)data.data(), offsets - (sizeof(float) + k_off), 1);
  Vec16f f011 =
      _mm512_i32gather_ps((float*)data.data(), offsets - k_off, 1);
  Vec16f f100 = _mm512_i32gather_ps(
      (float*)data.data(), offsets - (sizeof(float) + data.pitch()), 1);
  Vec16f f101 = _mm512_i32gather_ps((float*)data.data(),
                                   offsets - data.pitch(), 1);
  Vec16f f110 = _mm512_i32gather_ps((float*)data.data(),
                                   offsets - sizeof(float), 1);
  Vec16f f111 = _mm512_i32gather_ps((float*)data.data(), offsets, 1);

  f000 = simd::lerp(x3, f000, f100);
  f010 = simd::lerp(x3, f010, f110);
  f001 = simd::lerp(x3, f001, f101);
  f011 = simd::lerp(x3, f011, f111);

  f000 = simd::lerp(x2, f000, f010);
  f001 = simd::lerp(x2, f001, f011);

  return simd::lerp(x1, f000, f001);
}
#endif

}  // namespace Aperture

#endif  // _INTERPOLATION_H_
