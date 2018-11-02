#ifndef _PTC_UPDATER_HELPER_H_
#define _PTC_UPDATER_HELPER_H_

#include "data/typedefs.h"
#include "utils/interpolation.cuh"


namespace Aperture {

namespace Kernels {

typedef Spline::cloud_in_cell spline_t;

HD_INLINE Scalar
cloud_in_cell_f(Scalar dx) {
  return max(1.0 - std::abs(dx), 0.0);
}

HD_INLINE Scalar
interpolate(Scalar rel_pos, int ptc_cell, int target_cell) {
  Scalar dx =
      ((Scalar)target_cell + 0.5 - (rel_pos + Scalar(ptc_cell)));
  return cloud_in_cell_f(dx);
}

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

}


}

#endif