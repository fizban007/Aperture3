#include "algorithms/ffe_solver_logsph.h"
#include "cuda/constant_mem.h"
#include "cuda/cudaUtility.h"
#include "cuda/data_ptrs.h"
#include "cuda/grids/grid_log_sph_ptrs.h"
#include "cuda/utils/interpolation.cuh"
#include "cuda/utils/pitchptr.h"
#include "grids/grid_log_sph.h"
#include "sim_data.h"
#include "sim_environment.h"
#include "utils/timer.h"

const __device__ float TINY = 1e-7;

namespace Aperture {

static dim3 blockSize(32, 4, 4);
static dim3 blockGroupSize;
static int SHIFT_GHOST = 2;

namespace Kernels {

__global__ void
ffe_logsph_compute_rho(pitchptr<Scalar> e1, pitchptr<Scalar> e2,
                       pitchptr<Scalar> e3, pitchptr<Scalar> rho,
                       int shift) {
  size_t ijk;
  int i =
      threadIdx.x + blockIdx.x * blockDim.x + dev_mesh.guard[0] - shift;
  int j =
      threadIdx.y + blockIdx.y * blockDim.y + dev_mesh.guard[1] - shift;
  int k =
      threadIdx.z + blockIdx.z * blockDim.z + dev_mesh.guard[2] - shift;
  if (i < dev_mesh.dims[0] - dev_mesh.guard[0] + shift &&
      j < dev_mesh.dims[1] - dev_mesh.guard[1] + shift &&
      k < dev_mesh.dims[2] - dev_mesh.guard[2] + shift) {
    ijk = i + j * dev_mesh.dims[0] +
          k * dev_mesh.dims[0] * dev_mesh.dims[1];
    Scalar x = dev_mesh.pos(0, i, 1);
    Scalar r = exp(x);
    Scalar theta = dev_mesh.pos(1, j, 1);
    Scalar sth = sin(theta);
    rho[ijk] =
        dev_mesh.inv_delta[0] *
            (e1[ijk + 1] * exp(2.0 * (x + 0.5 * dev_mesh.delta[0])) -
             e1[ijk] * exp(2.0 * (x - 0.5 * dev_mesh.delta[0]))) /
            (r * r * r) +
        dev_mesh.inv_delta[1] *
            (e2[ijk + dev_mesh.dims[0]] *
                 sin(theta + 0.5 * dev_mesh.delta[1]) -
             e2[ijk] * sin(theta - 0.5 * dev_mesh.delta[1])) /
            (r * sth) +
        dev_mesh.inv_delta[2] *
            (e3[ijk + dev_mesh.dims[0] * dev_mesh.dims[1]] - e3[ijk]) /
            (r * sth);
  }
}

__global__ void
ffe_logsph_rk_push(pitchptr<Scalar> e1, pitchptr<Scalar> e2,
                   pitchptr<Scalar> e3, pitchptr<Scalar> b1,
                   pitchptr<Scalar> b2, pitchptr<Scalar> b3,
                   pitchptr<Scalar> b01, pitchptr<Scalar> b02,
                   pitchptr<Scalar> b03, pitchptr<Scalar> de1,
                   pitchptr<Scalar> de2, pitchptr<Scalar> de3,
                   pitchptr<Scalar> db1, pitchptr<Scalar> db2,
                   pitchptr<Scalar> db3, pitchptr<Scalar> rho,
                   int shift) {
  Scalar dtd1 = dev_params.delta_t * dev_mesh.inv_delta[0];
  Scalar dtd2 = dev_params.delta_t * dev_mesh.inv_delta[1];
  Scalar dtd3 = dev_params.delta_t * dev_mesh.inv_delta[2];
  size_t dj = dev_mesh.dims[0];
  size_t dk = dev_mesh.dims[0] * dev_mesh.dims[1];

  int i =
      threadIdx.x + blockIdx.x * blockDim.x + dev_mesh.guard[0] - shift;
  int j =
      threadIdx.y + blockIdx.y * blockDim.y + dev_mesh.guard[1] - shift;
  int k =
      threadIdx.z + blockIdx.z * blockDim.z + dev_mesh.guard[2] - shift;
  size_t ijk;
  Scalar r, theta;
  if (i < dev_mesh.dims[0] - dev_mesh.guard[0] + shift &&
      j < dev_mesh.dims[1] - dev_mesh.guard[1] + shift &&
      k < dev_mesh.dims[2] - dev_mesh.guard[2] + shift) {
    ijk = i + j * dev_mesh.dims[0] +
          k * dev_mesh.dims[0] * dev_mesh.dims[1];

    theta = dev_mesh.pos(1, j, 0);
    db1[ijk] =
        (dtd2 * (e3[ijk] * sin(theta + 0.5 * dev_mesh.delta[1]) -
                 e3[ijk - dj] * sin(theta - 0.5 * dev_mesh.delta[1])) -
         dtd3 * (e2[ijk] - e2[ijk - dk])) /
        (exp(dev_mesh.pos(0, i, 1)) * sin(theta));

    theta += 0.5 * dev_mesh.delta[1];
    r = exp(dev_mesh.pos(0, i, 0));
    db2[ijk] = (dtd3 * (e1[ijk] - e1[ijk - dk]) / sin(theta) -
                dtd1 *
                    (e3[ijk] * exp(dev_mesh.pos(0, i, 1)) -
                     e3[ijk - 1] * exp(dev_mesh.pos(0, i - 1, 1))) /
                    r) /
               r;

    db3[ijk] = (dtd1 *
                    (e2[ijk] * exp(dev_mesh.pos(0, i, 1)) -
                     e2[ijk - 1] * exp(dev_mesh.pos(0, i - 1, 1))) /
                    r -
                dtd2 * (e1[ijk] - e1[ijk - dj])) /
               r;

    de1[ijk] =
        (dtd2 * ((b3[ijk + dj] - b03[ijk + dj]) *
                     sin(theta + 0.5 * dev_mesh.delta[1]) -
                 (b3[ijk] - b03[ijk]) *
                     sin(theta - 0.5 * dev_mesh.delta[1])) -
         dtd3 * (b2[ijk + dk] - b2[ijk] - b02[ijk + dk] + b02[ijk])) /
        (r * sin(theta));

    theta -= 0.5 * dev_mesh.delta[1];
    r = exp(dev_mesh.pos(0, i, 1));
    de2[ijk] =
        (dtd3 * (b1[ijk + dk] - b1[ijk] - b01[ijk + dk] + b01[ijk]) /
             sin(theta) -
         dtd1 *
             ((b3[ijk + 1] - b03[ijk + 1]) *
                  exp(dev_mesh.pos(0, i + 1, 0)) -
              (b3[ijk] - b03[ijk]) * exp(dev_mesh.pos(0, i, 0))) /
             r) /
        r;

    de3[ijk] =
        (dtd1 *
             ((b2[ijk + 1] - b02[ijk + 1]) *
                  exp(dev_mesh.pos(0, i + 1, 0)) -
              (b2[ijk] - b02[ijk]) * exp(dev_mesh.pos(0, i, 0))) /
             r -
         dtd2 * (b1[ijk + dj] - b1[ijk] - b01[ijk + dj] + b01[ijk])) /
        r;

    Scalar intrho =
        interpolate(rho, ijk, Stagger(0b111), Stagger(0b110),
                    dev_mesh.dims[0], dev_mesh.dims[1]);
    Scalar intex = interpolate(e1, ijk, Stagger(0b110), Stagger(0b110),
                               dev_mesh.dims[0], dev_mesh.dims[1]);
    Scalar intey = interpolate(e2, ijk, Stagger(0b101), Stagger(0b110),
                               dev_mesh.dims[0], dev_mesh.dims[1]);
    Scalar intez = interpolate(e3, ijk, Stagger(0b011), Stagger(0b110),
                               dev_mesh.dims[0], dev_mesh.dims[1]);
    Scalar intbx = interpolate(b1, ijk, Stagger(0b001), Stagger(0b110),
                               dev_mesh.dims[0], dev_mesh.dims[1]);
    Scalar intby = interpolate(b2, ijk, Stagger(0b010), Stagger(0b110),
                               dev_mesh.dims[0], dev_mesh.dims[1]);
    Scalar intbz = interpolate(b3, ijk, Stagger(0b100), Stagger(0b110),
                               dev_mesh.dims[0], dev_mesh.dims[1]);
    Scalar jx = dev_params.delta_t * intrho *
                (intey * intbz - intez * intby) /
                (intbx * intbx + intby * intby + intbz * intbz + TINY);
    de1[ijk] -= jx;

    intrho = interpolate(rho, ijk, Stagger(0b111), Stagger(0b101),
                         dev_mesh.dims[0], dev_mesh.dims[1]);
    intex = interpolate(e1, ijk, Stagger(0b110), Stagger(0b101),
                        dev_mesh.dims[0], dev_mesh.dims[1]);
    intey = interpolate(e2, ijk, Stagger(0b101), Stagger(0b101),
                        dev_mesh.dims[0], dev_mesh.dims[1]);
    intez = interpolate(e3, ijk, Stagger(0b011), Stagger(0b101),
                        dev_mesh.dims[0], dev_mesh.dims[1]);
    intbx = interpolate(b1, ijk, Stagger(0b001), Stagger(0b101),
                        dev_mesh.dims[0], dev_mesh.dims[1]);
    intby = interpolate(b2, ijk, Stagger(0b010), Stagger(0b101),
                        dev_mesh.dims[0], dev_mesh.dims[1]);
    intbz = interpolate(b3, ijk, Stagger(0b100), Stagger(0b101),
                        dev_mesh.dims[0], dev_mesh.dims[1]);
    jx = dev_params.delta_t * intrho * (intez * intbx - intex * intbz) /
         (intbx * intbx + intby * intby + intbz * intbz + TINY);
    de2[ijk] -= jx;

    intrho = interpolate(rho, ijk, Stagger(0b111), Stagger(0b011),
                         dev_mesh.dims[0], dev_mesh.dims[1]);
    intex = interpolate(e1, ijk, Stagger(0b110), Stagger(0b011),
                        dev_mesh.dims[0], dev_mesh.dims[1]);
    intey = interpolate(e2, ijk, Stagger(0b101), Stagger(0b011),
                        dev_mesh.dims[0], dev_mesh.dims[1]);
    intez = interpolate(e3, ijk, Stagger(0b011), Stagger(0b011),
                        dev_mesh.dims[0], dev_mesh.dims[1]);
    intbx = interpolate(b1, ijk, Stagger(0b001), Stagger(0b011),
                        dev_mesh.dims[0], dev_mesh.dims[1]);
    intby = interpolate(b2, ijk, Stagger(0b010), Stagger(0b011),
                        dev_mesh.dims[0], dev_mesh.dims[1]);
    intbz = interpolate(b3, ijk, Stagger(0b100), Stagger(0b011),
                        dev_mesh.dims[0], dev_mesh.dims[1]);
    jx = dev_params.delta_t * intrho * (intex * intby - intey * intbx) /
         (intbx * intbx + intby * intby + intbz * intbz + TINY);
    de3[ijk] -= jx;
  }
}

__global__ void
ffe_rk_update(pitchptr<Scalar> e1, pitchptr<Scalar> e2,
              pitchptr<Scalar> e3, pitchptr<Scalar> b1,
              pitchptr<Scalar> b2, pitchptr<Scalar> b3,
              pitchptr<Scalar> en1, pitchptr<Scalar> en2,
              pitchptr<Scalar> en3, pitchptr<Scalar> bn1,
              pitchptr<Scalar> bn2, pitchptr<Scalar> bn3,
              pitchptr<Scalar> de1, pitchptr<Scalar> de2,
              pitchptr<Scalar> de3, pitchptr<Scalar> db1,
              pitchptr<Scalar> db2, pitchptr<Scalar> db3, Scalar rk_c1,
              Scalar rk_c2, Scalar rk_c3, int shift) {
  size_t ijk;
  int i =
      threadIdx.x + blockIdx.x * blockDim.x + dev_mesh.guard[0] - shift;
  int j =
      threadIdx.y + blockIdx.y * blockDim.y + dev_mesh.guard[1] - shift;
  int k =
      threadIdx.z + blockIdx.z * blockDim.z + dev_mesh.guard[2] - shift;
  if (i < dev_mesh.dims[0] - dev_mesh.guard[0] + shift &&
      j < dev_mesh.dims[1] - dev_mesh.guard[1] + shift &&
      k < dev_mesh.dims[2] - dev_mesh.guard[2] + shift) {
    ijk = i + j * dev_mesh.dims[0] +
          k * dev_mesh.dims[0] * dev_mesh.dims[1];
    // update E-field
    e1[ijk] = rk_c1 * en1[ijk] + rk_c2 * e1[ijk] + rk_c3 * de1[ijk];
    e2[ijk] = rk_c1 * en2[ijk] + rk_c2 * e2[ijk] + rk_c3 * de2[ijk];
    e3[ijk] = rk_c1 * en3[ijk] + rk_c2 * e3[ijk] + rk_c3 * de3[ijk];
    de1[ijk] = e1[ijk];
    de2[ijk] = e2[ijk];
    de3[ijk] = e3[ijk];
    // update B-field
    b1[ijk] = rk_c1 * bn1[ijk] + rk_c2 * b1[ijk] + rk_c3 * db1[ijk];
    b2[ijk] = rk_c1 * bn2[ijk] + rk_c2 * b2[ijk] + rk_c3 * db2[ijk];
    b3[ijk] = rk_c1 * bn3[ijk] + rk_c2 * b3[ijk] + rk_c3 * db3[ijk];
  }
}

__global__ void
ffe_clean_epar(pitchptr<Scalar> e1, pitchptr<Scalar> e2,
               pitchptr<Scalar> e3, pitchptr<Scalar> b1,
               pitchptr<Scalar> b2, pitchptr<Scalar> b3,
               pitchptr<Scalar> de1, pitchptr<Scalar> de2,
               pitchptr<Scalar> de3, int shift) {
  Scalar intex, intey, intez, intbx, intby, intbz;
  size_t ijk;
  int i =
      threadIdx.x + blockIdx.x * blockDim.x + dev_mesh.guard[0] - shift;
  int j =
      threadIdx.y + blockIdx.y * blockDim.y + dev_mesh.guard[1] - shift;
  int k =
      threadIdx.z + blockIdx.z * blockDim.z + dev_mesh.guard[2] - shift;
  if (i < dev_mesh.dims[0] - dev_mesh.guard[0] + shift &&
      j < dev_mesh.dims[1] - dev_mesh.guard[1] + shift &&
      k < dev_mesh.dims[2] - dev_mesh.guard[2] + shift) {
    ijk = i + j * dev_mesh.dims[0] +
          k * dev_mesh.dims[0] * dev_mesh.dims[1];
    // x:
    intex = interpolate(e1, ijk, Stagger(0b110), Stagger(0b110),
                        dev_mesh.dims[0], dev_mesh.dims[1]);
    intey = interpolate(e2, ijk, Stagger(0b101), Stagger(0b110),
                        dev_mesh.dims[0], dev_mesh.dims[1]);
    intez = interpolate(e3, ijk, Stagger(0b011), Stagger(0b110),
                        dev_mesh.dims[0], dev_mesh.dims[1]);
    intbx = interpolate(b1, ijk, Stagger(0b001), Stagger(0b110),
                        dev_mesh.dims[0], dev_mesh.dims[1]);
    intby = interpolate(b2, ijk, Stagger(0b010), Stagger(0b110),
                        dev_mesh.dims[0], dev_mesh.dims[1]);
    intbz = interpolate(b3, ijk, Stagger(0b100), Stagger(0b110),
                        dev_mesh.dims[0], dev_mesh.dims[1]);
    de1[ijk] =
        e1[ijk] -
        (intex * intbx + intey * intby + intez * intbz) * intbx /
            (intbx * intbx + intby * intby + intbz * intbz + TINY);

    // y:
    intex = interpolate(e1, ijk, Stagger(0b110), Stagger(0b101),
                        dev_mesh.dims[0], dev_mesh.dims[1]);
    intey = interpolate(e2, ijk, Stagger(0b101), Stagger(0b101),
                        dev_mesh.dims[0], dev_mesh.dims[1]);
    intez = interpolate(e3, ijk, Stagger(0b011), Stagger(0b101),
                        dev_mesh.dims[0], dev_mesh.dims[1]);
    intbx = interpolate(b1, ijk, Stagger(0b001), Stagger(0b101),
                        dev_mesh.dims[0], dev_mesh.dims[1]);
    intby = interpolate(b2, ijk, Stagger(0b010), Stagger(0b101),
                        dev_mesh.dims[0], dev_mesh.dims[1]);
    intbz = interpolate(b3, ijk, Stagger(0b100), Stagger(0b101),
                        dev_mesh.dims[0], dev_mesh.dims[1]);
    de2[ijk] =
        e2[ijk] -
        (intex * intbx + intey * intby + intez * intbz) * intby /
            (intbx * intbx + intby * intby + intbz * intbz + TINY);

    // z:
    intex = interpolate(e1, ijk, Stagger(0b110), Stagger(0b011),
                        dev_mesh.dims[0], dev_mesh.dims[1]);
    intey = interpolate(e2, ijk, Stagger(0b101), Stagger(0b011),
                        dev_mesh.dims[0], dev_mesh.dims[1]);
    intez = interpolate(e3, ijk, Stagger(0b011), Stagger(0b011),
                        dev_mesh.dims[0], dev_mesh.dims[1]);
    intbx = interpolate(b1, ijk, Stagger(0b001), Stagger(0b011),
                        dev_mesh.dims[0], dev_mesh.dims[1]);
    intby = interpolate(b2, ijk, Stagger(0b010), Stagger(0b011),
                        dev_mesh.dims[0], dev_mesh.dims[1]);
    intbz = interpolate(b3, ijk, Stagger(0b100), Stagger(0b011),
                        dev_mesh.dims[0], dev_mesh.dims[1]);
    de3[ijk] =
        e3[ijk] -
        (intex * intbx + intey * intby + intez * intbz) * intbz /
            (intbx * intbx + intby * intby + intbz * intbz + TINY);
  }
}

__global__ void
ffe_check_eGTb(pitchptr<Scalar> e1, pitchptr<Scalar> e2,
               pitchptr<Scalar> e3, pitchptr<Scalar> b1,
               pitchptr<Scalar> b2, pitchptr<Scalar> b3,
               pitchptr<Scalar> de1, pitchptr<Scalar> de2,
               pitchptr<Scalar> de3, int shift) {
  Scalar intex, intey, intez, intbx, intby, intbz, emag, bmag, temp;
  size_t ijk;
  int i =
      threadIdx.x + blockIdx.x * blockDim.x + dev_mesh.guard[0] - shift;
  int j =
      threadIdx.y + blockIdx.y * blockDim.y + dev_mesh.guard[1] - shift;
  int k =
      threadIdx.z + blockIdx.z * blockDim.z + dev_mesh.guard[2] - shift;
  if (i < dev_mesh.dims[0] - dev_mesh.guard[0] + shift &&
      j < dev_mesh.dims[1] - dev_mesh.guard[1] + shift &&
      k < dev_mesh.dims[2] - dev_mesh.guard[2] + shift) {
    ijk = i + j * dev_mesh.dims[0] +
          k * dev_mesh.dims[0] * dev_mesh.dims[1];
    // x:
    intex = interpolate(e1, ijk, Stagger(0b110), Stagger(0b110),
                        dev_mesh.dims[0], dev_mesh.dims[1]);
    intey = interpolate(e2, ijk, Stagger(0b101), Stagger(0b110),
                        dev_mesh.dims[0], dev_mesh.dims[1]);
    intez = interpolate(e3, ijk, Stagger(0b011), Stagger(0b110),
                        dev_mesh.dims[0], dev_mesh.dims[1]);
    emag = intex * intex + intey * intey + intez * intez + TINY;
    intbx = interpolate(b1, ijk, Stagger(0b001), Stagger(0b110),
                        dev_mesh.dims[0], dev_mesh.dims[1]);
    intby = interpolate(b2, ijk, Stagger(0b010), Stagger(0b110),
                        dev_mesh.dims[0], dev_mesh.dims[1]);
    intbz = interpolate(b3, ijk, Stagger(0b100), Stagger(0b110),
                        dev_mesh.dims[0], dev_mesh.dims[1]);
    bmag = intbx * intbx + intby * intby + intbz * intbz + TINY;
    if (emag > bmag) {
      temp = sqrt(bmag / emag);
    } else {
      temp = 1.0;
    }
    e1[ijk] = temp * de1[ijk];

    // y:
    intex = interpolate(e1, ijk, Stagger(0b110), Stagger(0b101),
                        dev_mesh.dims[0], dev_mesh.dims[1]);
    intey = interpolate(e2, ijk, Stagger(0b101), Stagger(0b101),
                        dev_mesh.dims[0], dev_mesh.dims[1]);
    intez = interpolate(e3, ijk, Stagger(0b011), Stagger(0b101),
                        dev_mesh.dims[0], dev_mesh.dims[1]);
    emag = intex * intex + intey * intey + intez * intez + TINY;
    intbx = interpolate(b1, ijk, Stagger(0b001), Stagger(0b101),
                        dev_mesh.dims[0], dev_mesh.dims[1]);
    intby = interpolate(b2, ijk, Stagger(0b010), Stagger(0b101),
                        dev_mesh.dims[0], dev_mesh.dims[1]);
    intbz = interpolate(b3, ijk, Stagger(0b100), Stagger(0b101),
                        dev_mesh.dims[0], dev_mesh.dims[1]);
    bmag = intbx * intbx + intby * intby + intbz * intbz + TINY;
    if (emag > bmag) {
      temp = sqrt(bmag / emag);
    } else {
      temp = 1.0;
    }
    e2[ijk] = temp * de2[ijk];

    // z:
    intex = interpolate(e1, ijk, Stagger(0b110), Stagger(0b011),
                        dev_mesh.dims[0], dev_mesh.dims[1]);
    intey = interpolate(e2, ijk, Stagger(0b101), Stagger(0b011),
                        dev_mesh.dims[0], dev_mesh.dims[1]);
    intez = interpolate(e3, ijk, Stagger(0b011), Stagger(0b011),
                        dev_mesh.dims[0], dev_mesh.dims[1]);
    emag = intex * intex + intey * intey + intez * intez + TINY;
    intbx = interpolate(b1, ijk, Stagger(0b001), Stagger(0b011),
                        dev_mesh.dims[0], dev_mesh.dims[1]);
    intby = interpolate(b2, ijk, Stagger(0b010), Stagger(0b011),
                        dev_mesh.dims[0], dev_mesh.dims[1]);
    intbz = interpolate(b3, ijk, Stagger(0b100), Stagger(0b011),
                        dev_mesh.dims[0], dev_mesh.dims[1]);
    bmag = intbx * intbx + intby * intby + intbz * intbz + TINY;
    if (emag > bmag) {
      temp = sqrt(bmag / emag);
    } else {
      temp = 1.0;
    }
    e3[ijk] = temp * de3[ijk];
  }
}

__global__ void
ffe_logsph_stellar_boundary(pitchptr<Scalar> e1, pitchptr<Scalar> e2,
                            pitchptr<Scalar> e3, pitchptr<Scalar> b1,
                            pitchptr<Scalar> b2, pitchptr<Scalar> b3,
                            Scalar omega) {
  for (int j = blockIdx.x * blockDim.x + threadIdx.x;
       j < dev_mesh.dims[1]; j += blockDim.x * gridDim.x) {
    Scalar theta_s = dev_mesh.pos(1, j, true);
    Scalar theta = dev_mesh.pos(1, j, false);
    // for (int i = 0; i < dev_mesh.guard[0] + 1; i++) {
    for (int i = 0; i < dev_mesh.guard[0]; i++) {
      Scalar r_s = std::exp(dev_mesh.pos(0, i, true));
      Scalar r = std::exp(dev_mesh.pos(0, i, false));
      Scalar omega_LT = 0.4f * omega * dev_params.compactness;
      b1(i, j) = 0.0f;
      e3(i, j) = 0.0f;
      e2(i, j) = -(omega - omega_LT) * std::sin(theta) *
                 dev_bg_fields.B1(i, j) / r_s / r_s;
      e1(i, j) = (omega - omega_LT) * std::sin(theta_s) *
                 dev_bg_fields.B2(i, j) / r / r;
      b2(i, j) = 0.0f;
      b3(i, j) = 0.0f;
    }
  }
}

__global__ void
ffe_logsph_axis_boundary_lower(pitchptr<Scalar> e1, pitchptr<Scalar> e2,
                               pitchptr<Scalar> e3, pitchptr<Scalar> b1,
                               pitchptr<Scalar> b2,
                               pitchptr<Scalar> b3) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < dev_mesh.dims[0]; i += blockDim.x * gridDim.x) {
    e3(i, dev_mesh.guard[1] - 1) = 0.0f;
    // e3(i, dev_mesh.guard[1]) = 0.0f;
    e2(i, dev_mesh.guard[1] - 1) = -e2(i, dev_mesh.guard[1]);
    // e2(i, dev_mesh.guard[1] - 1) = e2(i, dev_mesh.guard[1]) = 0.0f;

    b3(i, dev_mesh.guard[1] - 1) = b3(i, dev_mesh.guard[1]) = 0.0f;
    b2(i, dev_mesh.guard[1] - 1) = 0.0f;
    b1(i, dev_mesh.guard[1] - 1) = b1(i, dev_mesh.guard[1]);
  }
}

__global__ void
ffe_logsph_axis_boundary_upper(pitchptr<Scalar> e1, pitchptr<Scalar> e2,
                               pitchptr<Scalar> e3, pitchptr<Scalar> b1,
                               pitchptr<Scalar> b2,
                               pitchptr<Scalar> b3) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < dev_mesh.dims[0]; i += blockDim.x * gridDim.x) {
    int j_last = dev_mesh.dims[1] - dev_mesh.guard[1];
    e3(i, j_last - 1) = 0.0f;
    e2(i, j_last) = -e2(i, j_last - 1);
    // e2(i, j_last) = e2(i, j_last - 1) = 0.0f;

    b3(i, j_last) = b3(i, j_last - 1) = 0.0f;
    b2(i, j_last - 1) = 0.0f;
    b1(i, j_last) = b1(i, j_last - 1);
  }
}

__global__ void
ffe_logsph_outflow_boundary(pitchptr<Scalar> e1, pitchptr<Scalar> e2,
                            pitchptr<Scalar> e3, pitchptr<Scalar> b1,
                            pitchptr<Scalar> b2, pitchptr<Scalar> b3) {
  for (int j = blockIdx.x * blockDim.x + threadIdx.x;
       j < dev_mesh.dims[1]; j += blockDim.x * gridDim.x) {
    for (int i = 0; i < dev_params.damping_length; i++) {
      int n1 = dev_mesh.dims[0] - dev_params.damping_length + i;
      // size_t offset = j * e1.pitch + n1 * sizeof(Scalar);
      size_t offset = e1.compute_offset(n1, j);
      Scalar lambda =
          1.0f - dev_params.damping_coef *
                     square((Scalar)i / dev_params.damping_length);
      e1[offset] *= lambda;
      e2[offset] *= lambda;
      e3[offset] *= lambda;
      // b1[offset] *= lambda;
      // b2[offset] *= lambda;
      b3[offset] *= lambda;
    }
  }
}

}  // namespace Kernels

ffe_solver_logsph::ffe_solver_logsph(sim_environment &env)
    : m_env(env),
      En(env.grid()),
      dE(env.grid()),
      Bn(env.grid()),
      dB(env.grid()),
      rho(env.grid()) {
  En.set_stagger(0, Stagger(0b110));
  En.set_stagger(1, Stagger(0b101));
  En.set_stagger(2, Stagger(0b011));
  dE.set_stagger(0, Stagger(0b110));
  dE.set_stagger(1, Stagger(0b101));
  dE.set_stagger(2, Stagger(0b011));
  Bn.set_stagger(0, Stagger(0b001));
  Bn.set_stagger(1, Stagger(0b010));
  Bn.set_stagger(2, Stagger(0b100));
  dB.set_stagger(0, Stagger(0b001));
  dB.set_stagger(1, Stagger(0b010));
  dB.set_stagger(2, Stagger(0b100));

  En.initialize();
  dE.initialize();
  Bn.initialize();
  dB.initialize();

  rho.initialize();

  auto &mesh = env.grid().mesh();
  blockGroupSize =
      dim3((mesh.reduced_dim(0) + SHIFT_GHOST * 2 + blockSize.x - 1) /
               blockSize.x,
           (mesh.reduced_dim(1) + SHIFT_GHOST * 2 + blockSize.y - 1) /
               blockSize.y,
           (mesh.reduced_dim(2) + SHIFT_GHOST * 2 + blockSize.z - 1) /
               blockSize.z);
}

ffe_solver_logsph::~ffe_solver_logsph() {}

// void
// ffe_solver_logsph::update_fields(sim_data &data, double dt,
//                                  double time) {
//   // Only implemented 2D!
//   if (data.env.grid().dim() != 2) return;
//   timer::stamp("field_update");

//   // First communicate to get the E field guard cells
//   // data.env.get_sub_guard_cells(data.E);

//   Grid_LogSph &grid = *dynamic_cast<Grid_LogSph
//   *>(&m_env.local_grid()); auto mesh_ptrs = get_mesh_ptrs(grid); auto
//   &mesh = grid.mesh();

//   dim3 blockSize(32, 16);
//   dim3 gridSize(mesh.reduced_dim(0) / 32, mesh.reduced_dim(1) / 16);
//   // Update B
//   Kernels::compute_b_update<<<gridSize, blockSize>>>(
//       get_pitchptr(data.E.data(0)), get_pitchptr(data.E.data(1)),
//       get_pitchptr(data.E.data(2)), get_pitchptr(data.B.data(0)),
//       get_pitchptr(data.B.data(1)), get_pitchptr(data.B.data(2)),
//       mesh_ptrs, dt);
//   CudaCheckError();

//   // Communicate the new B values to guard cells
//   // data.env.get_sub_guard_cells(data.B);
//   // data.env.get_sub_guard_cells(data.J);

//   // Update E
//   Kernels::compute_e_update<<<gridSize, blockSize>>>(
//       get_pitchptr(data.E.data(0)), get_pitchptr(data.E.data(1)),
//       get_pitchptr(data.E.data(2)), get_pitchptr(data.B.data(0)),
//       get_pitchptr(data.B.data(1)), get_pitchptr(data.B.data(2)),
//       get_pitchptr(data.J.data(0)), get_pitchptr(data.J.data(1)),
//       get_pitchptr(data.J.data(2)), mesh_ptrs, dt);
//   CudaCheckError();

//   // Communicate the new E values to guard cells
//   // data.env.get_sub_guard_cells(data.E);

//   // Update B
//   Kernels::compute_divs<<<gridSize, blockSize>>>(
//       get_pitchptr(data.E.data(0)), get_pitchptr(data.E.data(1)),
//       get_pitchptr(data.E.data(2)), get_pitchptr(data.B.data(0)),
//       get_pitchptr(data.B.data(1)), get_pitchptr(data.B.data(2)),
//       get_pitchptr(data.divE.data()), get_pitchptr(data.divB.data()),
//       mesh_ptrs);
//   CudaCheckError();
//   data.compute_edotb();

//   CudaSafeCall(cudaDeviceSynchronize());
//   timer::show_duration_since_stamp("Field update", "us",
//                                    "field_update");
// }
void
ffe_solver_logsph::update_fields(sim_data &data, double dt,
                                 double time) {
  // RANGE_PUSH("Compute", CLR_GREEN);
  copy_fields(data);

  // substep #1:
  rk_push(data, dt);
  rk_update(data, 1.0, 0.0, 1.0);
  check_eGTb(data);
  CudaSafeCall(cudaDeviceSynchronize());
  // RANGE_POP;
  m_env.send_field_guard_cells(data);

  // substep #2:
  // RANGE_PUSH("Compute", CLR_GREEN);
  rk_push(data, dt);
  rk_update(data, 0.75, 0.25, 0.25);
  check_eGTb(data);
  CudaSafeCall(cudaDeviceSynchronize());
  // RANGE_POP;
  m_env.send_field_guard_cells(data);

  // substep #3:
  // RANGE_PUSH("Compute", CLR_GREEN);
  rk_push(data, dt);
  rk_update(data, 1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0);
  clean_epar(data);
  check_eGTb(data);
  CudaSafeCall(cudaDeviceSynchronize());
  // RANGE_POP;

  m_env.send_field_guard_cells(data);
}

void
ffe_solver_logsph::apply_boundary(sim_data &data, double omega,
                                  double time) {
  // int dev_id = data.dev_id;
  // CudaSafeCall(cudaSetDevice(dev_id));
  if (data.env.is_boundary(BoundaryPos::lower0)) {
    Kernels::ffe_logsph_stellar_boundary<<<32, 256>>>(
        get_pitchptr(data.E.data(0)), get_pitchptr(data.E.data(1)),
        get_pitchptr(data.E.data(2)), get_pitchptr(data.B.data(0)),
        get_pitchptr(data.B.data(1)), get_pitchptr(data.B.data(2)),
        omega);
    CudaCheckError();
  }

  if (data.env.is_boundary(BoundaryPos::upper0)) {
    Kernels::ffe_logsph_outflow_boundary<<<32, 256>>>(
        get_pitchptr(data.E.data(0)), get_pitchptr(data.E.data(1)),
        get_pitchptr(data.E.data(2)), get_pitchptr(data.B.data(0)),
        get_pitchptr(data.B.data(1)), get_pitchptr(data.B.data(2)));
    CudaCheckError();
  }

  if (data.env.is_boundary(BoundaryPos::lower1)) {
    Kernels::ffe_logsph_axis_boundary_lower<<<32, 256>>>(
        get_pitchptr(data.E.data(0)), get_pitchptr(data.E.data(1)),
        get_pitchptr(data.E.data(2)), get_pitchptr(data.B.data(0)),
        get_pitchptr(data.B.data(1)), get_pitchptr(data.B.data(2)));
    CudaCheckError();
  }

  if (data.env.is_boundary(BoundaryPos::upper1)) {
    Kernels::ffe_logsph_axis_boundary_upper<<<32, 256>>>(
        get_pitchptr(data.E.data(0)), get_pitchptr(data.E.data(1)),
        get_pitchptr(data.E.data(2)), get_pitchptr(data.B.data(0)),
        get_pitchptr(data.B.data(1)), get_pitchptr(data.B.data(2)));
    CudaCheckError();
  }
  // Logger::print_info("omega is {}", omega);
}

void
ffe_solver_logsph::copy_fields(const sim_data &data) {
  En.copy_from(data.E);
  Bn.copy_from(data.B);
  dE.initialize();
  dB.initialize();
}

void
ffe_solver_logsph::rk_push(sim_data &data, double dt) {
  Kernels::ffe_logsph_compute_rho<<<blockGroupSize, blockSize>>>(
      get_pitchptr(data.E.data(0)), get_pitchptr(data.E.data(1)),
      get_pitchptr(data.E.data(2)), get_pitchptr(rho.data()),
      SHIFT_GHOST);
  CudaCheckError();
  // `dE = curl B - curl B0 - j, dB = -curl E`
  // kernel_rk_push<<<g, blockSize>>>(
  Kernels::ffe_logsph_rk_push<<<blockGroupSize, blockSize>>>(
      get_pitchptr(data.E.data(0)), get_pitchptr(data.E.data(1)), get_pitchptr(data.E.data(2)),
      get_pitchptr(data.B.data(0)), get_pitchptr(data.B.data(1)), get_pitchptr(data.B.data(2)),
      get_pitchptr(data.Bbg.data(0)), get_pitchptr(data.Bbg.data(1)), get_pitchptr(data.Bbg.data(2)),
      get_pitchptr(dE.data(0)), get_pitchptr(dE.data(1)), get_pitchptr(dE.data(2)),
      get_pitchptr(dB.data(0)), get_pitchptr(dB.data(1)), get_pitchptr(dB.data(2)),
      get_pitchptr(rho.data()), SHIFT_GHOST);
  CudaCheckError();
}

void
ffe_solver_logsph::rk_update(sim_data &data, Scalar c1, Scalar c2, Scalar c3) {
  Kernels::ffe_rk_update<<<blockGroupSize, blockSize>>>(
      get_pitchptr(data.E.data(0)), get_pitchptr(data.E.data(1)), get_pitchptr(data.E.data(2)),
      get_pitchptr(data.B.data(0)), get_pitchptr(data.B.data(1)), get_pitchptr(data.B.data(2)),
      get_pitchptr(En.data(0)), get_pitchptr(En.data(1)), get_pitchptr(En.data(2)),
      get_pitchptr(Bn.data(0)), get_pitchptr(Bn.data(1)), get_pitchptr(Bn.data(2)),
      get_pitchptr(dE.data(0)), get_pitchptr(dE.data(1)), get_pitchptr(dE.data(2)),
      get_pitchptr(dB.data(0)), get_pitchptr(dB.data(1)), get_pitchptr(dB.data(2)),
      c1, c2, c3, SHIFT_GHOST);
  CudaCheckError();
}

void
ffe_solver_logsph::clean_epar(sim_data& data) {
  Kernels::ffe_clean_epar<<<blockGroupSize, blockSize>>>(
      get_pitchptr(data.E.data(0)), get_pitchptr(data.E.data(1)), get_pitchptr(data.E.data(2)),
      get_pitchptr(data.B.data(0)), get_pitchptr(data.B.data(1)), get_pitchptr(data.B.data(2)),
      get_pitchptr(dE.data(0)), get_pitchptr(dE.data(1)), get_pitchptr(dE.data(2)),
      SHIFT_GHOST);
  CudaCheckError();
}

void
ffe_solver_logsph::check_eGTb(sim_data& data) {
  Kernels::ffe_check_eGTb<<<blockGroupSize, blockSize>>>(
      get_pitchptr(data.E.data(0)), get_pitchptr(data.E.data(1)), get_pitchptr(data.E.data(2)),
      get_pitchptr(data.B.data(0)), get_pitchptr(data.B.data(1)), get_pitchptr(data.B.data(2)),
      get_pitchptr(dE.data(0)), get_pitchptr(dE.data(1)), get_pitchptr(dE.data(2)),
      SHIFT_GHOST);
  CudaCheckError();
}

}  // namespace Aperture
