#ifndef _AVX_INTERP_H_
#define _AVX_INTERP_H_

#include "constant_defs.h"
#include "data/multi_array.h"
#include "data/stagger.h"
#include "data/typedefs.h"
#include "data/vec3.h"
#include "utils/simd.h"

namespace Aperture {

template <typename Float, typename VF, typename VI>
VF
interpolate_3d(const multi_array<Float>& data, VI offsets, VF x1,
               VF x2, VF x3, Stagger stagger) {
  // auto empty_mask = (cells == Vec8ui(MAX_CELL));
  // VI d = select(empty_mask, Vec8ui(1 + data.height()),
  //               cells / Divisor_ui(data.width()));
  // VI c1s = select(empty_mask, Vec8ui(1), cells - d * data.width());
  // VI offsets = c1s * sizeof(float) + d * data.pitch();
  uint32_t k_off = data.pitch() * data.height();

  auto nx1 = select((bool)stagger[0], 0, truncate_to_int(x1 + 0.5));
  auto nx2 = select((bool)stagger[1], 0, truncate_to_int(x2 + 0.5));
  auto nx3 = select((bool)stagger[2], 0, truncate_to_int(x3 + 0.5));
  x1 = select((bool)stagger[0], x1, x1 + 0.5 - to_float(nx1));
  x2 = select((bool)stagger[1], x2, x2 + 0.5 - to_float(nx2));
  x3 = select((bool)stagger[2], x3, x3 + 0.5 - to_float(nx3));
  offsets += nx1 * sizeof(float);
  offsets += nx2 * data.pitch();
  offsets += nx3 * k_off;

  VF f000 = gather((float*)data.data(),
                   offsets - (k_off + sizeof(float) + data.pitch()), 1);
  VF f001 =
      gather((float*)data.data(), offsets - (k_off + data.pitch()), 1);
  VF f010 =
      gather((float*)data.data(), offsets - (sizeof(float) + k_off), 1);
  VF f011 =
      gather((float*)data.data(), offsets - k_off, 1);
  VF f100 = gather((float*)data.data(),
                   offsets - (sizeof(float) + data.pitch()), 1);
  VF f101 = gather((float*)data.data(), offsets - data.pitch(), 1);
  VF f110 = gather((float*)data.data(), offsets - sizeof(float), 1);
  VF f111 = gather((float*)data.data(), offsets, 1);

  f000 = simd::lerp(x3, f000, f100);
  f010 = simd::lerp(x3, f010, f110);
  f001 = simd::lerp(x3, f001, f101);
  f011 = simd::lerp(x3, f011, f111);

  f000 = simd::lerp(x2, f000, f010);
  f001 = simd::lerp(x2, f001, f011);

  return simd::lerp(x1, f000, f001);
}

template <typename VF>
inline VF
interp_1(VF dx) {
  auto abs_dx = abs(dx);
  return max(1.0 - abs_dx, VF(0.0));
}

template <>
inline float
interp_1<float>(float dx) {
  auto abs_dx = std::abs(dx);
  return std::max(1.0f - abs_dx, 0.0f);
}

}  // namespace Aperture

#endif  // _AVX_INTERP_H_
