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

constexpr HOST_DEVICE Scalar
B0() {
  return 1000.0;
}
constexpr HOST_DEVICE Scalar
p1() {
  return 0.0;
}
constexpr HOST_DEVICE Scalar
p2() {
  return 0.3;
}
constexpr HOST_DEVICE Scalar
p3() {
  return 0.03;
}
constexpr HOST_DEVICE Scalar
q11() {
  return -0.4;
}
constexpr HOST_DEVICE Scalar
q12() {
  return 0.0;
}
constexpr HOST_DEVICE Scalar
q13() {
  return 0.0;
}
constexpr HOST_DEVICE Scalar
q22() {
  return 0.5;
}
constexpr HOST_DEVICE Scalar
q23() {
  return -0.7;
}

HOST_DEVICE Scalar
star_field_b1(Scalar r, Scalar theta, Scalar phi) {
  Scalar cth = cos(theta);
  return 2.0 * B0() * cth / (r * r * r);
}

HOST_DEVICE Scalar
star_field_b2(Scalar r, Scalar theta, Scalar phi) {
  Scalar sth = sin(theta);
  return B0() * sth / (r * r * r);
}

HOST_DEVICE Scalar
star_field_b3(Scalar r, Scalar theta, Scalar phi) {
  return 0.0;
}

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
    size_t di = sizeof(Scalar);
    size_t dj = e1.p.pitch;
    size_t dk = e1.p.pitch * dev_mesh.dims[1];
    ijk = i * di + j * dj + k * dk;
    Scalar x = dev_mesh.pos(0, i, 1);
    Scalar r = exp(x);
    Scalar theta = dev_mesh.pos(1, j, 1);
    Scalar sth = sin(theta);
    rho[ijk] =
        dev_mesh.inv_delta[0] *
            (e1[ijk + di] * exp(2.0 * (x + 0.5 * dev_mesh.delta[0])) -
             e1[ijk] * exp(2.0 * (x - 0.5 * dev_mesh.delta[0]))) /
            (r * r * r) +
        dev_mesh.inv_delta[1] *
            (e2[ijk + dj] * sin(theta + 0.5 * dev_mesh.delta[1]) -
             e2[ijk] * sin(theta - 0.5 * dev_mesh.delta[1])) /
            (r * sth) +
        dev_mesh.inv_delta[2] * (e3[ijk + dk] - e3[ijk]) / (r * sth);
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
  size_t di = sizeof(Scalar);
  size_t dj = e1.p.pitch;
  size_t dk = e1.p.pitch * dev_mesh.dims[1];

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
    ijk = i * di + j * dj + k * dk;

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
                     e3[ijk - di] * exp(dev_mesh.pos(0, i - 1, 1))) /
                    r) /
               r;

    db3[ijk] = (dtd1 *
                    (e2[ijk] * exp(dev_mesh.pos(0, i, 1)) -
                     e2[ijk - di] * exp(dev_mesh.pos(0, i - 1, 1))) /
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
             ((b3[ijk + di] - b03[ijk + di]) *
                  exp(dev_mesh.pos(0, i + 1, 0)) -
              (b3[ijk] - b03[ijk]) * exp(dev_mesh.pos(0, i, 0))) /
             r) /
        r;

    de3[ijk] =
        (dtd1 *
             ((b2[ijk + di] - b02[ijk + di]) *
                  exp(dev_mesh.pos(0, i + 1, 0)) -
              (b2[ijk] - b02[ijk]) * exp(dev_mesh.pos(0, i, 0))) /
             r -
         dtd2 * (b1[ijk + dj] - b1[ijk] - b01[ijk + dj] + b01[ijk])) /
        r;

    // Scalar intrho =
    //     interpolate(rho, ijk, Stagger(0b111), Stagger(0b110),
    //                 e1.p.pitch, dev_mesh.dims[1]);
    // Scalar intex = interpolate(e1, ijk, Stagger(0b110),
    // Stagger(0b110),
    //                            e1.p.pitch, dev_mesh.dims[1]);
    // Scalar intey = interpolate(e2, ijk, Stagger(0b101),
    // Stagger(0b110),
    //                            e1.p.pitch, dev_mesh.dims[1]);
    // Scalar intez = interpolate(e3, ijk, Stagger(0b011),
    // Stagger(0b110),
    //                            e1.p.pitch, dev_mesh.dims[1]);
    // Scalar intbx = interpolate(b1, ijk, Stagger(0b001),
    // Stagger(0b110),
    //                            e1.p.pitch, dev_mesh.dims[1]);
    // Scalar intby = interpolate(b2, ijk, Stagger(0b010),
    // Stagger(0b110),
    //                            e1.p.pitch, dev_mesh.dims[1]);
    // Scalar intbz = interpolate(b3, ijk, Stagger(0b100),
    // Stagger(0b110),
    //                            e1.p.pitch, dev_mesh.dims[1]);
    // Scalar jx = dev_params.delta_t * intrho *
    //             (intey * intbz - intez * intby) /
    //             (intbx * intbx + intby * intby + intbz * intbz +
    //             TINY);
    // de1[ijk] -= jx;

    // intrho = interpolate(rho, ijk, Stagger(0b111), Stagger(0b101),
    //                      e1.p.pitch, dev_mesh.dims[1]);
    // intex = interpolate(e1, ijk, Stagger(0b110), Stagger(0b101),
    //                     e1.p.pitch, dev_mesh.dims[1]);
    // intey = interpolate(e2, ijk, Stagger(0b101), Stagger(0b101),
    //                     e1.p.pitch, dev_mesh.dims[1]);
    // intez = interpolate(e3, ijk, Stagger(0b011), Stagger(0b101),
    //                     e1.p.pitch, dev_mesh.dims[1]);
    // intbx = interpolate(b1, ijk, Stagger(0b001), Stagger(0b101),
    //                     e1.p.pitch, dev_mesh.dims[1]);
    // intby = interpolate(b2, ijk, Stagger(0b010), Stagger(0b101),
    //                     e1.p.pitch, dev_mesh.dims[1]);
    // intbz = interpolate(b3, ijk, Stagger(0b100), Stagger(0b101),
    //                     e1.p.pitch, dev_mesh.dims[1]);
    // jx = dev_params.delta_t * intrho * (intez * intbx - intex *
    // intbz) /
    //      (intbx * intbx + intby * intby + intbz * intbz + TINY);
    // de2[ijk] -= jx;

    // intrho = interpolate(rho, ijk, Stagger(0b111), Stagger(0b011),
    //                      e1.p.pitch, dev_mesh.dims[1]);
    // intex = interpolate(e1, ijk, Stagger(0b110), Stagger(0b011),
    //                     e1.p.pitch, dev_mesh.dims[1]);
    // intey = interpolate(e2, ijk, Stagger(0b101), Stagger(0b011),
    //                     e1.p.pitch, dev_mesh.dims[1]);
    // intez = interpolate(e3, ijk, Stagger(0b011), Stagger(0b011),
    //                     e1.p.pitch, dev_mesh.dims[1]);
    // intbx = interpolate(b1, ijk, Stagger(0b001), Stagger(0b011),
    //                     e1.p.pitch, dev_mesh.dims[1]);
    // intby = interpolate(b2, ijk, Stagger(0b010), Stagger(0b011),
    //                     e1.p.pitch, dev_mesh.dims[1]);
    // intbz = interpolate(b3, ijk, Stagger(0b100), Stagger(0b011),
    //                     e1.p.pitch, dev_mesh.dims[1]);
    // jx = dev_params.delta_t * intrho * (intex * intby - intey *
    // intbx) /
    //      (intbx * intbx + intby * intby + intbz * intbz + TINY);
    // de3[ijk] -= jx;
  }
}

__global__ void
ffe_logsph_compute_j(pitchptr<Scalar> e1, pitchptr<Scalar> e2,
                     pitchptr<Scalar> e3, pitchptr<Scalar> b1,
                     pitchptr<Scalar> b2, pitchptr<Scalar> b3,
                     pitchptr<Scalar> j1, pitchptr<Scalar> j2,
                     pitchptr<Scalar> j3, pitchptr<Scalar> rho,
                     int shift) {
  size_t di = sizeof(Scalar);
  size_t dj = e1.p.pitch;
  size_t dk = e1.p.pitch * dev_mesh.dims[1];

  int i =
      threadIdx.x + blockIdx.x * blockDim.x + dev_mesh.guard[0] - shift;
  int j =
      threadIdx.y + blockIdx.y * blockDim.y + dev_mesh.guard[1] - shift;
  int k =
      threadIdx.z + blockIdx.z * blockDim.z + dev_mesh.guard[2] - shift;
  size_t ijk;
  if (i < dev_mesh.dims[0] - dev_mesh.guard[0] + shift &&
      j < dev_mesh.dims[1] - dev_mesh.guard[1] + shift &&
      k < dev_mesh.dims[2] - dev_mesh.guard[2] + shift) {
    ijk = i * di + j * dj + k * dk;
    Scalar intrho =
        interpolate(rho, ijk, Stagger(0b111), Stagger(0b110),
                    e1.p.pitch, dev_mesh.dims[1]);
    Scalar intex = interpolate(e1, ijk, Stagger(0b110), Stagger(0b110),
                               e1.p.pitch, dev_mesh.dims[1]);
    Scalar intey = interpolate(e2, ijk, Stagger(0b101), Stagger(0b110),
                               e2.p.pitch, dev_mesh.dims[1]);
    Scalar intez = interpolate(e3, ijk, Stagger(0b011), Stagger(0b110),
                               e3.p.pitch, dev_mesh.dims[1]);
    Scalar intbx = interpolate(b1, ijk, Stagger(0b001), Stagger(0b110),
                               b1.p.pitch, dev_mesh.dims[1]);
    Scalar intby = interpolate(b2, ijk, Stagger(0b010), Stagger(0b110),
                               b2.p.pitch, dev_mesh.dims[1]);
    Scalar intbz = interpolate(b3, ijk, Stagger(0b100), Stagger(0b110),
                               b3.p.pitch, dev_mesh.dims[1]);
    Scalar jx = dev_params.delta_t * intrho *
                (intey * intbz - intez * intby) /
                (intbx * intbx + intby * intby + intbz * intbz + TINY);
    j1[ijk] = jx;

    intrho = interpolate(rho, ijk, Stagger(0b111), Stagger(0b101),
                         rho.p.pitch, dev_mesh.dims[1]);
    intex = interpolate(e1, ijk, Stagger(0b110), Stagger(0b101),
                        e1.p.pitch, dev_mesh.dims[1]);
    intey = interpolate(e2, ijk, Stagger(0b101), Stagger(0b101),
                        e2.p.pitch, dev_mesh.dims[1]);
    intez = interpolate(e3, ijk, Stagger(0b011), Stagger(0b101),
                        e3.p.pitch, dev_mesh.dims[1]);
    intbx = interpolate(b1, ijk, Stagger(0b001), Stagger(0b101),
                        b1.p.pitch, dev_mesh.dims[1]);
    intby = interpolate(b2, ijk, Stagger(0b010), Stagger(0b101),
                        b2.p.pitch, dev_mesh.dims[1]);
    intbz = interpolate(b3, ijk, Stagger(0b100), Stagger(0b101),
                        b3.p.pitch, dev_mesh.dims[1]);
    jx = dev_params.delta_t * intrho * (intez * intbx - intex * intbz) /
         (intbx * intbx + intby * intby + intbz * intbz + TINY);
    j2[ijk] = jx;

    intrho = interpolate(rho, ijk, Stagger(0b111), Stagger(0b011),
                         rho.p.pitch, dev_mesh.dims[1]);
    intex = interpolate(e1, ijk, Stagger(0b110), Stagger(0b011),
                        e1.p.pitch, dev_mesh.dims[1]);
    intey = interpolate(e2, ijk, Stagger(0b101), Stagger(0b011),
                        e2.p.pitch, dev_mesh.dims[1]);
    intez = interpolate(e3, ijk, Stagger(0b011), Stagger(0b011),
                        e3.p.pitch, dev_mesh.dims[1]);
    intbx = interpolate(b1, ijk, Stagger(0b001), Stagger(0b011),
                        b1.p.pitch, dev_mesh.dims[1]);
    intby = interpolate(b2, ijk, Stagger(0b010), Stagger(0b011),
                        b2.p.pitch, dev_mesh.dims[1]);
    intbz = interpolate(b3, ijk, Stagger(0b100), Stagger(0b011),
                        b3.p.pitch, dev_mesh.dims[1]);
    jx = dev_params.delta_t * intrho * (intex * intby - intey * intbx) /
         (intbx * intbx + intby * intby + intbz * intbz + TINY);
    j3[ijk] = jx;
  }
}

__global__ void
ffe_logsph_rk_update_b(pitchptr<Scalar> db1, pitchptr<Scalar> db2,
                       pitchptr<Scalar> db3, pitchptr<Scalar> e1,
                       pitchptr<Scalar> e2, pitchptr<Scalar> e3,
                       mesh_ptrs_log_sph mesh_ptrs, int shift) {
  Scalar dt = dev_params.delta_t;
  size_t di = sizeof(Scalar);
  size_t dj = e1.p.pitch;
  size_t dk = e1.p.pitch * dev_mesh.dims[1];

  int i =
      threadIdx.x + blockIdx.x * blockDim.x + dev_mesh.guard[0] - shift;
  int j =
      threadIdx.y + blockIdx.y * blockDim.y + dev_mesh.guard[1] - shift;
  int k =
      threadIdx.z + blockIdx.z * blockDim.z + dev_mesh.guard[2] - shift;
  size_t ijk, ij;
  if (i < dev_mesh.dims[0] - dev_mesh.guard[0] + shift &&
      j < dev_mesh.dims[1] - dev_mesh.guard[1] + shift &&
      k < dev_mesh.dims[2] - dev_mesh.guard[2] + shift) {
    ij = i * di + j * dj;
    ijk = ij + k * dk;

    db1[ijk] = dt *
               -(e3[ijk] * mesh_ptrs.l3_e[ij] -
                 e3[ijk - dj] * mesh_ptrs.l3_e[ij - dj] -
                 (e2[ijk] - e2[ijk - dk]) * mesh_ptrs.l2_e[ij]) /
               mesh_ptrs.A1_b[ij];

    db2[ijk] = dt *
               -((e1[ijk] - e1[ijk - dk]) * mesh_ptrs.l1_e[ij] -
                 e3[ijk] * mesh_ptrs.l3_e[ij] +
                 e3[ijk - di] * mesh_ptrs.l3_e[ij - di]) /
               mesh_ptrs.A2_b[ij];

    db3[ijk] = dt *
               -(e2[ijk] * mesh_ptrs.l2_e[ij] -
                 e2[ijk - di] * mesh_ptrs.l2_e[ij - di] -
                 e1[ijk] * mesh_ptrs.l1_e[ij] +
                 e1[ijk - dj] * mesh_ptrs.l1_e[ij - dj]) /
               mesh_ptrs.A3_b[ij];
  }
}

__global__ void
ffe_logsph_rk_update_e(pitchptr<Scalar> de1, pitchptr<Scalar> de2,
                       pitchptr<Scalar> de3, pitchptr<Scalar> b1,
                       pitchptr<Scalar> b2, pitchptr<Scalar> b3,
                       pitchptr<Scalar> b01, pitchptr<Scalar> b02,
                       pitchptr<Scalar> b03, pitchptr<Scalar> j1,
                       pitchptr<Scalar> j2, pitchptr<Scalar> j3,
                       mesh_ptrs_log_sph mesh_ptrs, int shift) {
  Scalar dt = dev_params.delta_t;
  size_t di = sizeof(Scalar);
  size_t dj = b1.p.pitch;
  size_t dk = b1.p.pitch * dev_mesh.dims[1];

  int i =
      threadIdx.x + blockIdx.x * blockDim.x + dev_mesh.guard[0] - shift;
  int j =
      threadIdx.y + blockIdx.y * blockDim.y + dev_mesh.guard[1] - shift;
  int k =
      threadIdx.z + blockIdx.z * blockDim.z + dev_mesh.guard[2] - shift;
  size_t ijk, ij;
  if (i < dev_mesh.dims[0] - dev_mesh.guard[0] + shift &&
      j < dev_mesh.dims[1] - dev_mesh.guard[1] + shift &&
      k < dev_mesh.dims[2] - dev_mesh.guard[2] + shift) {
    ij = i * di + j * dj;
    ijk = ij + k * dk;

    de1[ijk] =
        dt *
            ((b3[ijk + dj] - b03[ijk + dj]) * mesh_ptrs.l3_b[ij + dj] -
             (b3[ijk] - b03[ijk]) * mesh_ptrs.l3_b[ij] -
             (b2[ijk + dk] - b02[ijk + dk]) * mesh_ptrs.l2_b[ij] +
             (b2[ijk] - b02[ijk]) * mesh_ptrs.l2_b[ij]) /
            mesh_ptrs.A1_e[ij] -
        dt * j1[ijk];

    de2[ijk] =
        dt *
            ((b1[ijk + dk] - b01[ijk + dk] - b1[ijk] + b01[ijk]) *
                 mesh_ptrs.l1_b[ij] -
             (b3[ijk + di] - b03[ijk + di]) * mesh_ptrs.l3_b[ij + di] +
             (b3[ijk] - b03[ijk]) * mesh_ptrs.l3_b[ij]) /
            mesh_ptrs.A2_e[ij] -
        dt * j2[ijk];

    de3[ijk] =
        dt *
            ((b2[ijk + di] - b02[ijk + di]) * mesh_ptrs.l2_b[ij + di] -
             (b2[ijk] - b02[ijk]) * mesh_ptrs.l2_b[ij] -
             (b1[ijk + dj] - b01[ijk + dj]) * mesh_ptrs.l1_b[ij + dj] +
             (b1[ijk] - b01[ijk]) * mesh_ptrs.l1_b[ij]) /
            mesh_ptrs.A3_e[ij] -
        dt * j3[ijk];

    // Scalar intrho =
    //     interpolate(rho, ijk, Stagger(0b111), Stagger(0b110),
    //                 e1.p.pitch, dev_mesh.dims[1]);
    // Scalar intex = interpolate(e1, ijk, Stagger(0b110),
    // Stagger(0b110),
    //                            e1.p.pitch, dev_mesh.dims[1]);
    // Scalar intey = interpolate(e2, ijk, Stagger(0b101),
    // Stagger(0b110),
    //                            e1.p.pitch, dev_mesh.dims[1]);
    // Scalar intez = interpolate(e3, ijk, Stagger(0b011),
    // Stagger(0b110),
    //                            e1.p.pitch, dev_mesh.dims[1]);
    // Scalar intbx = interpolate(b1, ijk, Stagger(0b001),
    // Stagger(0b110),
    //                            e1.p.pitch, dev_mesh.dims[1]);
    // Scalar intby = interpolate(b2, ijk, Stagger(0b010),
    // Stagger(0b110),
    //                            e1.p.pitch, dev_mesh.dims[1]);
    // Scalar intbz = interpolate(b3, ijk, Stagger(0b100),
    // Stagger(0b110),
    //                            e1.p.pitch, dev_mesh.dims[1]);
    // Scalar jx = dev_params.delta_t * intrho *
    //             (intey * intbz - intez * intby) /
    //             (intbx * intbx + intby * intby + intbz * intbz +
    //             TINY);
    // de1[ijk] -= jx;

    // intrho = interpolate(rho, ijk, Stagger(0b111), Stagger(0b101),
    //                      e1.p.pitch, dev_mesh.dims[1]);
    // intex = interpolate(e1, ijk, Stagger(0b110), Stagger(0b101),
    //                     e1.p.pitch, dev_mesh.dims[1]);
    // intey = interpolate(e2, ijk, Stagger(0b101), Stagger(0b101),
    //                     e1.p.pitch, dev_mesh.dims[1]);
    // intez = interpolate(e3, ijk, Stagger(0b011), Stagger(0b101),
    //                     e1.p.pitch, dev_mesh.dims[1]);
    // intbx = interpolate(b1, ijk, Stagger(0b001), Stagger(0b101),
    //                     e1.p.pitch, dev_mesh.dims[1]);
    // intby = interpolate(b2, ijk, Stagger(0b010), Stagger(0b101),
    //                     e1.p.pitch, dev_mesh.dims[1]);
    // intbz = interpolate(b3, ijk, Stagger(0b100), Stagger(0b101),
    //                     e1.p.pitch, dev_mesh.dims[1]);
    // jx = dev_params.delta_t * intrho * (intez * intbx - intex *
    // intbz) /
    //      (intbx * intbx + intby * intby + intbz * intbz + TINY);
    // de2[ijk] -= jx;

    // intrho = interpolate(rho, ijk, Stagger(0b111), Stagger(0b011),
    //                      e1.p.pitch, dev_mesh.dims[1]);
    // intex = interpolate(e1, ijk, Stagger(0b110), Stagger(0b011),
    //                     e1.p.pitch, dev_mesh.dims[1]);
    // intey = interpolate(e2, ijk, Stagger(0b101), Stagger(0b011),
    //                     e1.p.pitch, dev_mesh.dims[1]);
    // intez = interpolate(e3, ijk, Stagger(0b011), Stagger(0b011),
    //                     e1.p.pitch, dev_mesh.dims[1]);
    // intbx = interpolate(b1, ijk, Stagger(0b001), Stagger(0b011),
    //                     e1.p.pitch, dev_mesh.dims[1]);
    // intby = interpolate(b2, ijk, Stagger(0b010), Stagger(0b011),
    //                     e1.p.pitch, dev_mesh.dims[1]);
    // intbz = interpolate(b3, ijk, Stagger(0b100), Stagger(0b011),
    //                     e1.p.pitch, dev_mesh.dims[1]);
    // jx = dev_params.delta_t * intrho * (intex * intby - intey *
    // intbx) /
    //      (intbx * intbx + intby * intby + intbz * intbz + TINY);
    // de3[ijk] -= jx;
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
  size_t dj = e1.p.pitch;
  size_t dk = e1.p.pitch * dev_mesh.dims[1];
  if (i < dev_mesh.dims[0] - dev_mesh.guard[0] + shift &&
      j < dev_mesh.dims[1] - dev_mesh.guard[1] + shift &&
      k < dev_mesh.dims[2] - dev_mesh.guard[2] + shift) {
    ijk = i * sizeof(Scalar) + j * dj + k * dk;
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
  size_t dj = e1.p.pitch;
  size_t dk = e1.p.pitch * dev_mesh.dims[1];
  int i =
      threadIdx.x + blockIdx.x * blockDim.x + dev_mesh.guard[0] - shift;
  int j =
      threadIdx.y + blockIdx.y * blockDim.y + dev_mesh.guard[1] - shift;
  int k =
      threadIdx.z + blockIdx.z * blockDim.z + dev_mesh.guard[2] - shift;
  if (i < dev_mesh.dims[0] - dev_mesh.guard[0] + shift &&
      j < dev_mesh.dims[1] - dev_mesh.guard[1] + shift &&
      k < dev_mesh.dims[2] - dev_mesh.guard[2] + shift) {
    ijk = i * sizeof(Scalar) + j * dj + k * dk;
    // x:
    intex = interpolate(e1, ijk, Stagger(0b110), Stagger(0b110),
                        e1.p.pitch, dev_mesh.dims[1]);
    intey = interpolate(e2, ijk, Stagger(0b101), Stagger(0b110),
                        e1.p.pitch, dev_mesh.dims[1]);
    intez = interpolate(e3, ijk, Stagger(0b011), Stagger(0b110),
                        e1.p.pitch, dev_mesh.dims[1]);
    intbx = interpolate(b1, ijk, Stagger(0b001), Stagger(0b110),
                        e1.p.pitch, dev_mesh.dims[1]);
    intby = interpolate(b2, ijk, Stagger(0b010), Stagger(0b110),
                        e1.p.pitch, dev_mesh.dims[1]);
    intbz = interpolate(b3, ijk, Stagger(0b100), Stagger(0b110),
                        e1.p.pitch, dev_mesh.dims[1]);
    de1[ijk] =
        e1[ijk] -
        (intex * intbx + intey * intby + intez * intbz) * intbx /
            (intbx * intbx + intby * intby + intbz * intbz + TINY);

    // y:
    intex = interpolate(e1, ijk, Stagger(0b110), Stagger(0b101),
                        e1.p.pitch, dev_mesh.dims[1]);
    intey = interpolate(e2, ijk, Stagger(0b101), Stagger(0b101),
                        e1.p.pitch, dev_mesh.dims[1]);
    intez = interpolate(e3, ijk, Stagger(0b011), Stagger(0b101),
                        e1.p.pitch, dev_mesh.dims[1]);
    intbx = interpolate(b1, ijk, Stagger(0b001), Stagger(0b101),
                        e1.p.pitch, dev_mesh.dims[1]);
    intby = interpolate(b2, ijk, Stagger(0b010), Stagger(0b101),
                        e1.p.pitch, dev_mesh.dims[1]);
    intbz = interpolate(b3, ijk, Stagger(0b100), Stagger(0b101),
                        e1.p.pitch, dev_mesh.dims[1]);
    de2[ijk] =
        e2[ijk] -
        (intex * intbx + intey * intby + intez * intbz) * intby /
            (intbx * intbx + intby * intby + intbz * intbz + TINY);

    // z:
    intex = interpolate(e1, ijk, Stagger(0b110), Stagger(0b011),
                        e1.p.pitch, dev_mesh.dims[1]);
    intey = interpolate(e2, ijk, Stagger(0b101), Stagger(0b011),
                        e1.p.pitch, dev_mesh.dims[1]);
    intez = interpolate(e3, ijk, Stagger(0b011), Stagger(0b011),
                        e1.p.pitch, dev_mesh.dims[1]);
    intbx = interpolate(b1, ijk, Stagger(0b001), Stagger(0b011),
                        e1.p.pitch, dev_mesh.dims[1]);
    intby = interpolate(b2, ijk, Stagger(0b010), Stagger(0b011),
                        e1.p.pitch, dev_mesh.dims[1]);
    intbz = interpolate(b3, ijk, Stagger(0b100), Stagger(0b011),
                        e1.p.pitch, dev_mesh.dims[1]);
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
  size_t dj = e1.p.pitch;
  size_t dk = e1.p.pitch * dev_mesh.dims[1];
  int i =
      threadIdx.x + blockIdx.x * blockDim.x + dev_mesh.guard[0] - shift;
  int j =
      threadIdx.y + blockIdx.y * blockDim.y + dev_mesh.guard[1] - shift;
  int k =
      threadIdx.z + blockIdx.z * blockDim.z + dev_mesh.guard[2] - shift;
  if (i < dev_mesh.dims[0] - dev_mesh.guard[0] + shift &&
      j < dev_mesh.dims[1] - dev_mesh.guard[1] + shift &&
      k < dev_mesh.dims[2] - dev_mesh.guard[2] + shift) {
    ijk = i * sizeof(Scalar) + j * dj + k * dk;
    // x:
    intex = interpolate(e1, ijk, Stagger(0b110), Stagger(0b110),
                        e1.p.pitch, dev_mesh.dims[1]);
    intey = interpolate(e2, ijk, Stagger(0b101), Stagger(0b110),
                        e1.p.pitch, dev_mesh.dims[1]);
    intez = interpolate(e3, ijk, Stagger(0b011), Stagger(0b110),
                        e1.p.pitch, dev_mesh.dims[1]);
    emag = intex * intex + intey * intey + intez * intez + TINY;
    intbx = interpolate(b1, ijk, Stagger(0b001), Stagger(0b110),
                        e1.p.pitch, dev_mesh.dims[1]);
    intby = interpolate(b2, ijk, Stagger(0b010), Stagger(0b110),
                        e1.p.pitch, dev_mesh.dims[1]);
    intbz = interpolate(b3, ijk, Stagger(0b100), Stagger(0b110),
                        e1.p.pitch, dev_mesh.dims[1]);
    bmag = intbx * intbx + intby * intby + intbz * intbz + TINY;
    if (emag > bmag) {
      temp = sqrt(bmag / emag);
    } else {
      temp = 1.0;
    }
    e1[ijk] = temp * de1[ijk];

    // y:
    intex = interpolate(e1, ijk, Stagger(0b110), Stagger(0b101),
                        e1.p.pitch, dev_mesh.dims[1]);
    intey = interpolate(e2, ijk, Stagger(0b101), Stagger(0b101),
                        e1.p.pitch, dev_mesh.dims[1]);
    intez = interpolate(e3, ijk, Stagger(0b011), Stagger(0b101),
                        e1.p.pitch, dev_mesh.dims[1]);
    emag = intex * intex + intey * intey + intez * intez + TINY;
    intbx = interpolate(b1, ijk, Stagger(0b001), Stagger(0b101),
                        e1.p.pitch, dev_mesh.dims[1]);
    intby = interpolate(b2, ijk, Stagger(0b010), Stagger(0b101),
                        e1.p.pitch, dev_mesh.dims[1]);
    intbz = interpolate(b3, ijk, Stagger(0b100), Stagger(0b101),
                        e1.p.pitch, dev_mesh.dims[1]);
    bmag = intbx * intbx + intby * intby + intbz * intbz + TINY;
    if (emag > bmag) {
      temp = sqrt(bmag / emag);
    } else {
      temp = 1.0;
    }
    e2[ijk] = temp * de2[ijk];

    // z:
    intex = interpolate(e1, ijk, Stagger(0b110), Stagger(0b011),
                        e1.p.pitch, dev_mesh.dims[1]);
    intey = interpolate(e2, ijk, Stagger(0b101), Stagger(0b011),
                        e1.p.pitch, dev_mesh.dims[1]);
    intez = interpolate(e3, ijk, Stagger(0b011), Stagger(0b011),
                        e1.p.pitch, dev_mesh.dims[1]);
    emag = intex * intex + intey * intey + intez * intez + TINY;
    intbx = interpolate(b1, ijk, Stagger(0b001), Stagger(0b011),
                        e1.p.pitch, dev_mesh.dims[1]);
    intby = interpolate(b2, ijk, Stagger(0b010), Stagger(0b011),
                        e1.p.pitch, dev_mesh.dims[1]);
    intbz = interpolate(b3, ijk, Stagger(0b100), Stagger(0b011),
                        e1.p.pitch, dev_mesh.dims[1]);
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
                            pitchptr<Scalar> b01, pitchptr<Scalar> b02,
                            pitchptr<Scalar> b03, Scalar t,
                            Scalar omega) {
  int j = threadIdx.x + blockIdx.x * blockDim.x + dev_mesh.guard[1] - 1;
  int k = threadIdx.y + blockIdx.y * blockDim.y + dev_mesh.guard[2] - 1;
  if (j < dev_mesh.dims[1] - dev_mesh.guard[1] &&
      k < dev_mesh.dims[2] - dev_mesh.guard[2]) {
    Scalar theta_s = dev_mesh.pos(1, j, true);
    Scalar theta = dev_mesh.pos(1, j, false);
    Scalar phi_s = dev_mesh.pos(2, k, true);
    Scalar phi = dev_mesh.pos(2, k, false);
    // for (int i = 0; i < dev_mesh.guard[0] + 1; i++) {
    for (int i = 0; i < dev_mesh.guard[0]; i++) {
      Scalar r_s = std::exp(dev_mesh.pos(0, i, true));
      Scalar r = std::exp(dev_mesh.pos(0, i, false));
      // Scalar omega_LT = 0.4f * omega * dev_params.compactness;
      size_t ijk = i * sizeof(Scalar) + j * e1.p.pitch +
                   k * e1.p.pitch * dev_mesh.dims[1];
      b1[ijk] = star_field_b1(r_s, theta, phi + omega * t);
      b01[ijk] = star_field_b1(r_s, theta, phi + omega * t);
      e3[ijk] = 0.0f;
      // e2(i, j) = -(omega - omega_LT) * std::sin(theta) *
      //            dev_bg_fields.B1(i, j) / r_s / r_s;
      // e1(i, j) = (omega - omega_LT) * std::sin(theta_s) *
      //            dev_bg_fields.B2(i, j) / r / r;
      e2[ijk] = -omega * std::sin(theta) *
                star_field_b1(r_s, theta, phi_s + omega * t) * r_s;
      e1[ijk] = omega * std::sin(theta_s) *
                star_field_b2(r, theta_s, phi_s + omega * t) * r;
      b2[ijk] = star_field_b2(r, theta_s, phi + omega * t);
      b02[ijk] = star_field_b2(r, theta_s, phi + omega * t);
      b3[ijk] = star_field_b3(r, theta, phi_s + omega * t);
      b03[ijk] = star_field_b3(r, theta, phi_s + omega * t);
    }
  }
}

__global__ void
ffe_logsph_axis_boundary_lower(pitchptr<Scalar> e1, pitchptr<Scalar> e2,
                               pitchptr<Scalar> e3, pitchptr<Scalar> b1,
                               pitchptr<Scalar> b2, pitchptr<Scalar> b3,
                               pitchptr<Scalar> b01,
                               pitchptr<Scalar> b02,
                               pitchptr<Scalar> b03) {
  int i = threadIdx.x + blockIdx.x * blockDim.x + dev_mesh.guard[0];
  int k = threadIdx.y + blockIdx.y * blockDim.y + dev_mesh.guard[2];
  if (i < dev_mesh.dims[0] - dev_mesh.guard[0] &&
      k < dev_mesh.dims[2] - dev_mesh.guard[2]) {
    e1(i, dev_mesh.guard[1] - 1, k) = 0.0f;
    e3(i, dev_mesh.guard[1] - 1, k) = 0.0f;
    // e3(i, dev_mesh.guard[1]) = 0.0f;
    e2(i, dev_mesh.guard[1] - 1, k) = -e2(i, dev_mesh.guard[1], k);
    // e2(i, dev_mesh.guard[1] - 1) = e2(i, dev_mesh.guard[1]) = 0.0f;

    b3(i, dev_mesh.guard[1] - 1, k) = 0.0f;
    // b2(i, dev_mesh.guard[1] - 1, k) = b2(i, dev_mesh.guard[1], k);
    // b2(i, dev_mesh.guard[1] - 1, k) = b02(i, dev_mesh.guard[1] - 1, k);
    b2(i, dev_mesh.guard[1] - 1, k) = 0.0f;
    b02(i, dev_mesh.guard[1] - 1, k) = 0.0f;
    b1(i, dev_mesh.guard[1] - 1, k) = b1(i, dev_mesh.guard[1], k);
    // e1(i, dev_mesh.guard[1] - 1, k) = e1(i, dev_mesh.guard[1], k);
  }
}

__global__ void
ffe_logsph_axis_boundary_upper(pitchptr<Scalar> e1, pitchptr<Scalar> e2,
                               pitchptr<Scalar> e3, pitchptr<Scalar> b1,
                               pitchptr<Scalar> b2, pitchptr<Scalar> b3,
                               pitchptr<Scalar> b01,
                               pitchptr<Scalar> b02,
                               pitchptr<Scalar> b03) {
  int i = threadIdx.x + blockIdx.x * blockDim.x + dev_mesh.guard[0];
  int k = threadIdx.y + blockIdx.y * blockDim.y + dev_mesh.guard[2];
  if (i < dev_mesh.dims[0] - dev_mesh.guard[0] &&
      k < dev_mesh.dims[2] - dev_mesh.guard[2]) {
    int j_last = dev_mesh.dims[1] - dev_mesh.guard[1];
    e1(i, j_last - 1, k) = 0.0f;
    e3(i, j_last - 1, k) = 0.0f;
    e2(i, j_last, k) = -e2(i, j_last - 1, k);
    // e2(i, j_last, k) = 0.0f;

    b3(i, j_last - 1, k) = 0.0f;
    // b2(i, j_last - 1, k) = b02(i, j_last - 1, k);
    b2(i, j_last - 1, k) = b02(i, j_last - 1, k) = 0.0f;
    b1(i, j_last, k) = b1(i, j_last - 1, k);
  }
}

__global__ void
ffe_logsph_outflow_boundary(pitchptr<Scalar> e1, pitchptr<Scalar> e2,
                            pitchptr<Scalar> e3, pitchptr<Scalar> b1,
                            pitchptr<Scalar> b2, pitchptr<Scalar> b3) {
  int j = threadIdx.x + blockIdx.x * blockDim.x + dev_mesh.guard[1] - 1;
  int k = threadIdx.y + blockIdx.y * blockDim.y + dev_mesh.guard[2] - 1;
  if (j < dev_mesh.dims[1] - dev_mesh.guard[1] &&
      k < dev_mesh.dims[2] - dev_mesh.guard[2]) {
    for (int i = 0; i < dev_params.damping_length; i++) {
      int n1 = dev_mesh.dims[0] - dev_params.damping_length + i;
      // size_t offset = j * e1.pitch + n1 * sizeof(Scalar);
      size_t offset = e1.compute_offset(n1, j, k);
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
      J(env.grid()),
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
  J.initialize();

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

void
ffe_solver_logsph::update_fields(sim_data &data, double dt,
                                 double omega, double time) {
  // RANGE_PUSH("Compute", CLR_GREEN);
  copy_fields(data);

  // substep #1:
  timer::stamp();
  rk_push(data, dt);
  rk_update(data, 1.0, 0.0, 1.0);
  apply_boundary(data, omega, time + dt);
  // check_eGTb(data);
  CudaSafeCall(cudaDeviceSynchronize());
  // RANGE_POP;
  m_env.send_field_guard_cells(data);
  timer::show_duration_since_stamp("rk first substep", "ms");

  // substep #2:
  timer::stamp();
  // RANGE_PUSH("Compute", CLR_GREEN);
  rk_push(data, dt);
  rk_update(data, 0.75, 0.25, 0.25);
  apply_boundary(data, omega, time + 0.5 * dt);
  // check_eGTb(data);
  CudaSafeCall(cudaDeviceSynchronize());
  // RANGE_POP;
  m_env.send_field_guard_cells(data);
  timer::show_duration_since_stamp("rk second substep", "ms");

  // substep #3:
  timer::stamp();
  // RANGE_PUSH("Compute", CLR_GREEN);
  rk_push(data, dt);
  rk_update(data, 1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0);
  apply_boundary(data, omega, time + dt);
  // clean_epar(data);
  // check_eGTb(data);
  CudaSafeCall(cudaDeviceSynchronize());
  // RANGE_POP;

  m_env.send_field_guard_cells(data);
  timer::show_duration_since_stamp("rk third substep", "ms");
}

void
ffe_solver_logsph::apply_boundary(sim_data &data, double omega,
                                  double time) {
  auto &mesh = m_env.grid().mesh();
  // int dev_id = data.dev_id;
  // CudaSafeCall(cudaSetDevice(dev_id));
  if (data.env.is_boundary(BoundaryPos::lower0)) {
    dim3 gridSize((mesh.reduced_dim(1) + 1 + 31) / 32,
                  (mesh.reduced_dim(2) + 1 + 15) / 16);
    Kernels::ffe_logsph_stellar_boundary<<<gridSize, dim3(32, 16)>>>(
        get_pitchptr(data.E.data(0)), get_pitchptr(data.E.data(1)),
        get_pitchptr(data.E.data(2)), get_pitchptr(data.B.data(0)),
        get_pitchptr(data.B.data(1)), get_pitchptr(data.B.data(2)),
        get_pitchptr(data.Bbg.data(0)), get_pitchptr(data.Bbg.data(1)),
        get_pitchptr(data.Bbg.data(2)), time, omega);
    CudaCheckError();
  }

  if (data.env.is_boundary(BoundaryPos::upper0)) {
    dim3 gridSize((mesh.reduced_dim(1) + 1 + 31) / 32,
                  (mesh.reduced_dim(2) + 1 + 15) / 16);
    Kernels::ffe_logsph_outflow_boundary<<<gridSize, dim3(32, 16)>>>(
        get_pitchptr(data.E.data(0)), get_pitchptr(data.E.data(1)),
        get_pitchptr(data.E.data(2)), get_pitchptr(data.B.data(0)),
        get_pitchptr(data.B.data(1)), get_pitchptr(data.B.data(2)));
    CudaCheckError();
  }

  if (data.env.is_boundary(BoundaryPos::lower1)) {
    Kernels::ffe_logsph_axis_boundary_lower<<<32, 256>>>(
        get_pitchptr(data.E.data(0)), get_pitchptr(data.E.data(1)),
        get_pitchptr(data.E.data(2)), get_pitchptr(data.B.data(0)),
        get_pitchptr(data.B.data(1)), get_pitchptr(data.B.data(2)),
        get_pitchptr(data.Bbg.data(0)), get_pitchptr(data.Bbg.data(1)),
        get_pitchptr(data.Bbg.data(2)));
    CudaCheckError();
  }

  if (data.env.is_boundary(BoundaryPos::upper1)) {
    Kernels::ffe_logsph_axis_boundary_upper<<<32, 256>>>(
        get_pitchptr(data.E.data(0)), get_pitchptr(data.E.data(1)),
        get_pitchptr(data.E.data(2)), get_pitchptr(data.B.data(0)),
        get_pitchptr(data.B.data(1)), get_pitchptr(data.B.data(2)),
        get_pitchptr(data.Bbg.data(0)), get_pitchptr(data.Bbg.data(1)),
        get_pitchptr(data.Bbg.data(2)));
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
  // Kernels::ffe_logsph_compute_rho<<<blockGroupSize, blockSize>>>(
  //     get_pitchptr(data.E.data(0)), get_pitchptr(data.E.data(1)),
  //     get_pitchptr(data.E.data(2)), get_pitchptr(rho.data()),
  //     SHIFT_GHOST);
  // CudaCheckError();

  Grid_LogSph &grid = *dynamic_cast<Grid_LogSph *>(&m_env.local_grid());
  auto mesh_ptrs = get_mesh_ptrs(grid);

  // `dE = curl B - curl B0 - j, dB = -curl E`
  // kernel_rk_push<<<g, blockSize>>>(
  // Kernels::ffe_logsph_rk_push<<<blockGroupSize, blockSize>>>(
  //     get_pitchptr(data.E.data(0)), get_pitchptr(data.E.data(1)),
  //     get_pitchptr(data.E.data(2)), get_pitchptr(data.B.data(0)),
  //     get_pitchptr(data.B.data(1)), get_pitchptr(data.B.data(2)),
  //     get_pitchptr(data.Bbg.data(0)), get_pitchptr(data.Bbg.data(1)),
  //     get_pitchptr(data.Bbg.data(2)), get_pitchptr(dE.data(0)),
  //     get_pitchptr(dE.data(1)), get_pitchptr(dE.data(2)),
  //     get_pitchptr(dB.data(0)), get_pitchptr(dB.data(1)),
  //     get_pitchptr(dB.data(2)), get_pitchptr(rho.data()),
  //     SHIFT_GHOST);

  // Kernels::ffe_logsph_compute_j<<<blockGroupSize, blockSize>>>(
  //     get_pitchptr(data.E.data(0)), get_pitchptr(data.E.data(1)),
  //     get_pitchptr(data.E.data(2)), get_pitchptr(data.B.data(0)),
  //     get_pitchptr(data.B.data(1)), get_pitchptr(data.B.data(2)),
  //     get_pitchptr(J.data(0)), get_pitchptr(J.data(1)),
  //     get_pitchptr(J.data(2)), get_pitchptr(rho.data(2)),
  //     SHIFT_GHOST);
  // CudaCheckError();
  Kernels::ffe_logsph_rk_update_b<<<blockGroupSize, blockSize>>>(
      get_pitchptr(dB.data(0)), get_pitchptr(dB.data(1)),
      get_pitchptr(dB.data(2)), get_pitchptr(data.E.data(0)),
      get_pitchptr(data.E.data(1)), get_pitchptr(data.E.data(2)),
      mesh_ptrs, SHIFT_GHOST);
  CudaCheckError();

  Kernels::ffe_logsph_rk_update_e<<<blockGroupSize, blockSize>>>(
      get_pitchptr(dE.data(0)), get_pitchptr(dE.data(1)),
      get_pitchptr(dE.data(2)), get_pitchptr(data.B.data(0)),
      get_pitchptr(data.B.data(1)), get_pitchptr(data.B.data(2)),
      get_pitchptr(data.Bbg.data(0)), get_pitchptr(data.Bbg.data(1)),
      get_pitchptr(data.Bbg.data(2)), get_pitchptr(J.data(0)),
      get_pitchptr(J.data(1)), get_pitchptr(J.data(2)), mesh_ptrs,
      SHIFT_GHOST);
  CudaCheckError();
}

void
ffe_solver_logsph::rk_update(sim_data &data, Scalar c1, Scalar c2,
                             Scalar c3) {
  Kernels::ffe_rk_update<<<blockGroupSize, blockSize>>>(
      get_pitchptr(data.E.data(0)), get_pitchptr(data.E.data(1)),
      get_pitchptr(data.E.data(2)), get_pitchptr(data.B.data(0)),
      get_pitchptr(data.B.data(1)), get_pitchptr(data.B.data(2)),
      get_pitchptr(En.data(0)), get_pitchptr(En.data(1)),
      get_pitchptr(En.data(2)), get_pitchptr(Bn.data(0)),
      get_pitchptr(Bn.data(1)), get_pitchptr(Bn.data(2)),
      get_pitchptr(dE.data(0)), get_pitchptr(dE.data(1)),
      get_pitchptr(dE.data(2)), get_pitchptr(dB.data(0)),
      get_pitchptr(dB.data(1)), get_pitchptr(dB.data(2)), c1, c2, c3,
      SHIFT_GHOST);
  CudaCheckError();
}

void
ffe_solver_logsph::clean_epar(sim_data &data) {
  Kernels::ffe_clean_epar<<<blockGroupSize, blockSize>>>(
      get_pitchptr(data.E.data(0)), get_pitchptr(data.E.data(1)),
      get_pitchptr(data.E.data(2)), get_pitchptr(data.B.data(0)),
      get_pitchptr(data.B.data(1)), get_pitchptr(data.B.data(2)),
      get_pitchptr(dE.data(0)), get_pitchptr(dE.data(1)),
      get_pitchptr(dE.data(2)), SHIFT_GHOST);
  CudaCheckError();
}

void
ffe_solver_logsph::check_eGTb(sim_data &data) {
  Kernels::ffe_check_eGTb<<<blockGroupSize, blockSize>>>(
      get_pitchptr(data.E.data(0)), get_pitchptr(data.E.data(1)),
      get_pitchptr(data.E.data(2)), get_pitchptr(data.B.data(0)),
      get_pitchptr(data.B.data(1)), get_pitchptr(data.B.data(2)),
      get_pitchptr(dE.data(0)), get_pitchptr(dE.data(1)),
      get_pitchptr(dE.data(2)), SHIFT_GHOST);
  CudaCheckError();
}

}  // namespace Aperture
