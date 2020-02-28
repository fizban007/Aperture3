#ifndef _INTERPOLATION_H_
#define _INTERPOLATION_H_

#include "core/stagger.h"
#include "core/typedefs.h"
#include "cuda/constant_mem.h"
#include "cuda/cuda_control.h"
// #include "cuda/ptr_util.h"
#include "cuda/utils/pitchptr.h"
#include "utils/util_functions.h"

namespace Aperture {

namespace Spline {

template <int N>
struct spline_t;

template<>
struct spline_t<0> {
  enum { radius = 1, support = 1 };

  HD_INLINE Scalar operator()(Scalar dx) const {
    return (std::abs(dx) <= 0.5f ? 1.0f : 0.0f);
  }
};

template<>
struct spline_t<1> {
  enum { radius = 1, support = 2 };

  HD_INLINE Scalar operator()(Scalar dx) const {
    Scalar abs_dx = std::abs(dx);
    // return (abs_dx < 0.5 ? max(1.0 - abs_dx, 0.0) : abs_dx);
    return max(1.0f - abs_dx, 0.0f);
  }
};

template<>
struct spline_t<2> {
  enum { radius = 2, support = 4 };

  HD_INLINE Scalar operator()(Scalar dx) const {
    Scalar abs_dx = std::abs(dx);
    if (abs_dx < 0.5f) {
      return 0.75f - dx * dx;
    } else if (abs_dx < 1.5f) {
      Scalar tmp = 1.5f - abs_dx;
      return 0.5f * tmp * tmp;
    } else {
      return 0.0f;
    }
  }
};

template<>
struct spline_t<3> {
  enum { radius = 2, support = 4 };

  HD_INLINE Scalar operator()(Scalar dx) const {
    Scalar abs_dx = std::abs(dx);
    if (abs_dx < 1.0f) {
      Scalar tmp = abs_dx * abs_dx;
      return 0.6666667f - tmp + 0.5f * tmp * abs_dx;
    } else if (abs_dx < 2.0f) {
      Scalar tmp = 2.0f - abs_dx;
      return 0.1666667f * tmp * tmp * tmp;
    } else {
      return 0.0f;
    }
  }
};

using nearest_grid_point = spline_t<0>;
using cloud_in_cell = spline_t<1>;
using triangular_shaped_cloud = spline_t<2>;
using piecewise_cubic = spline_t<3>;

template <typename Interp, typename FloatT>
Scalar HD_INLINE
interp_cell(const Interp& interp, FloatT rel_pos, int c, int t,
            int stagger = 0) {
  // The actual distance between particle and t
  FloatT x = ((Scalar)t + (stagger == 0 ? 0.5f : 1.0f)) -
             (rel_pos + (Scalar)c);
  return interp(x);
}

}  // namespace Spline

template <typename Interp>
struct Interpolator3D {
  Interp interp;

  template <typename FloatT>
  HOST_DEVICE Scalar operator()(pitchptr<Scalar> f, FloatT x1,
                                FloatT x2, FloatT x3, int c1, int c2,
                                int c3, Stagger stagger) const {
    Scalar result = 0.0f;
    // for (int k = c3 - Interp::radius;
    //      k <= c3 + Interp::support - Interp::radius; k++) {
    for (int k = 0; k <= Interp::support; k++) {
      int kk = k + c3 - Interp::radius;
      size_t k_offset = kk * f.p.pitch * f.p.ysize;
      for (int j = 0; j <= Interp::support; j++) {
        // for (int j = c2 - Interp::radius;
        //      j <= c2 + Interp::support - Interp::radius; j++) {
        int jj = j + c2 - Interp::radius;
        size_t j_offset = jj * f.p.pitch;
        for (int i = 0; i <= Interp::support; i++) {
          // for (int i = c1 - Interp::radius;
          //      i <= c1 + Interp::support - Interp::radius; i++) {
          int ii = i + c1 - Interp::radius;
          size_t globalOffset =
              k_offset + j_offset + ii * sizeof(Scalar);

          result += f[globalOffset] *
                    interp_cell(interp, x1, c1, ii, stagger[0]) *
                    interp_cell(interp, x2, c2, jj, stagger[1]) *
                    interp_cell(interp, x3, c3, kk, stagger[2]);
        }
      }
    }
    return result;
  }

  template <typename FloatT>
  HOST_DEVICE Scalar compute_weight(FloatT x1, FloatT x2, FloatT x3,
                                    int c1, int c2, int c3, int t1,
                                    int t2, int t3,
                                    Stagger stagger) const {
    return interp_cell(interp, x1, c1, t1, stagger[0]) *
           interp_cell(interp, x2, c2, t2, stagger[1]) *
           interp_cell(interp, x3, c3, t3, stagger[2]);
  }

  template <typename FloatT>
  HD_INLINE Scalar interpolate(FloatT x) const {
    return interp(x);
  }

  HD_INLINE int radius() const { return Interp::radius + 1; }
  HD_INLINE int support() const { return Interp::support; }
};

template <typename Interp>
struct Interpolator2D {
  Interp interp;

  template <typename FloatT>
  HOST_DEVICE Scalar operator()(pitchptr<Scalar> f, FloatT x1,
                                FloatT x2, int c1, int c2,
                                Stagger stagger) const {
    Scalar result = 0.0f;
    for (int j = 0; j <= Interp::support; j++) {
      int jj = j + c2 - Interp::radius;
      size_t j_offset = jj * f.p.pitch;

      for (int i = 0; i <= Interp::support; i++) {
        int ii = i + c1 - Interp::radius;
        size_t globalOffset = j_offset + ii * sizeof(Scalar);
        // printf("add %f\n", *ptrAddr(f, globalOffset));

        result += f[globalOffset] *
                  interp_cell(interp, x1, c1, ii, stagger[0]) *
                  interp_cell(interp, x2, c2, jj, stagger[1]);
        // printf("%f\n",result);
      }
    }
    return result;
  }

  template <typename FloatT>
  HOST_DEVICE Scalar compute_weight(FloatT x1, FloatT x2, FloatT x3,
                                    int c1, int c2, int c3, int t1,
                                    int t2, int t3,
                                    Stagger stagger) const {
    return interp_cell(interp, x1, c1, t1, stagger[0]) *
           interp_cell(interp, x2, c2, t2, stagger[1]);
  }

  template <typename FloatT>
  HD_INLINE Scalar interpolate(FloatT x) const {
    return interp(x);
  }

  HD_INLINE int radius() const { return Interp::radius + 1; }
  HD_INLINE int support() const { return Interp::support; }
};

template <typename T>
HOST_DEVICE T
interpolate(pitchptr<T> f, std::size_t idx_lin, Stagger in, Stagger out,
            size_t dim0, size_t dim1) {
  int di_m = (in[0] == out[0] ? 0 : -1 - out[0]) * sizeof(T);
  int di_p = (in[0] == out[0] ? 0 : -out[0]) * sizeof(T);
  int dj_m = (in[1] == out[1] ? 0 : -1 - out[1]);
  int dj_p = (in[1] == out[1] ? 0 : -out[1]);
  int dk_m = (in[2] == out[2] ? 0 : -1 - out[2]);
  int dk_p = (in[2] == out[2] ? 0 : -out[2]);

  Scalar f11 =
      0.5 * (f[idx_lin + di_p + dj_p * dim0 + dk_m * dim0 * dim1] +
             f[idx_lin + di_p + dj_p * dim0 + dk_p * dim0 * dim1]);
  Scalar f10 =
      0.5 * (f[idx_lin + di_p + dj_m * dim0 + dk_m * dim0 * dim1] +
             f[idx_lin + di_p + dj_m * dim0 + dk_p * dim0 * dim1]);
  Scalar f01 =
      0.5 * (f[idx_lin + di_m + dj_p * dim0 + dk_m * dim0 * dim1] +
             f[idx_lin + di_m + dj_p * dim0 + dk_p * dim0 * dim1]);
  Scalar f00 =
      0.5 * (f[idx_lin + di_m + dj_m * dim0 + dk_m * dim0 * dim1] +
             f[idx_lin + di_m + dj_m * dim0 + dk_p * dim0 * dim1]);
  Scalar f1 = 0.5 * (f11 + f10);
  Scalar f0 = 0.5 * (f01 + f00);
  return 0.5 * (f1 + f0);
}

template <typename T>
HOST_DEVICE T
interpolate2d(pitchptr<T> f, std::size_t idx_lin, Stagger in, Stagger out,
              size_t dim0) {
  int di_m = (in[0] == out[0] ? 0 : -1 - out[0]) * sizeof(T);
  int di_p = (in[0] == out[0] ? 0 : -out[0]) * sizeof(T);
  int dj_m = (in[1] == out[1] ? 0 : -1 - out[1]);
  int dj_p = (in[1] == out[1] ? 0 : -out[1]);

  Scalar f1 = 0.5 * (f[idx_lin + di_p + dj_p * dim0] +
                     f[idx_lin + di_p + dj_m * dim0]);
  Scalar f0 = 0.5 * (f[idx_lin + di_m + dj_p * dim0] +
                     f[idx_lin + di_m + dj_m * dim0]);
  return 0.5 * (f1 + f0);
}

template <typename T>
HOST_DEVICE T
interpolate1d(pitchptr<T> f, std::size_t idx_lin, Stagger in, Stagger out,
              size_t dim0) {
  int di_m = (in[0] == out[0] ? 0 : -1 - out[0]) * sizeof(T);
  int di_p = (in[0] == out[0] ? 0 : -out[0]) * sizeof(T);

  Scalar f1 = f[idx_lin + di_p];
  Scalar f0 = f[idx_lin + di_m];
  return 0.5 * (f1 + f0);
}

// template <typename T>
// HOST_DEVICE T
// interpolate(pitchptr<T> f, Index idx, Stagger in, Stagger out, int dim0,
//             int dim1) {
//   std::size_t idx_lin = idx.x + idx.y * dim0 + idx.z * dim0 * dim1;
//   return interpolate(f, idx_lin, in, out, dim0, dim1);
// }

}  // namespace Aperture

#endif  // _INTERPOLATION_H_
