#include "algorithms/field_solver_logsph.h"
#include "cuda/constant_mem.h"
#include "cuda/cudaUtility.h"
#include "cuda/data_ptrs.h"
#include "cuda/grids/grid_log_sph_ptrs.h"
#include "cuda/utils/pitchptr.h"
#include "grids/grid_log_sph.h"
#include "sim_data.h"
#include "sim_environment.h"
#include "utils/double_buffer.h"
#include "utils/timer.h"

namespace Aperture {

namespace Kernels {

__global__ void filter_current_logsph(
    pitchptr<Scalar> j, pitchptr<Scalar> j_tmp, pitchptr<Scalar> A,
    bool boundary_lower0, bool boundary_upper0, bool boundary_lower1,
    bool boundary_upper1);

__device__ Scalar
beta_phi(Scalar r, Scalar theta) {
  // return -0.4f * dev_params.compactness * dev_params.omega *
  //        std::sin(theta) / (r * r);
  return 0.0f;
}

__device__ Scalar
alpha_gr(Scalar r) {
  // return std::sqrt(1.0f - dev_params.compactness / r);
  return 1.0f;
}

// template <int DIM1, int DIM2>
__global__ void
compute_e_update_logsph(data_ptrs data, mesh_ptrs_log_sph mesh_ptrs,
                        Scalar dt) {
  // Load position parameters
  int t1 = blockIdx.x, t2 = blockIdx.y;
  int c1 = threadIdx.x, c2 = threadIdx.y;
  int n1 = dev_mesh.guard[0] + t1 * blockDim.x + c1;
  int n2 = dev_mesh.guard[1] + t2 * blockDim.y + c2;
  size_t globalOffset = data.E1.compute_offset(n1, n2);

  Scalar r = std::exp(dev_mesh.pos(0, n1, true));
  Scalar theta = dev_mesh.pos(1, n2, true);
  Scalar theta0 = dev_mesh.pos(1, n2, false);
  // Scalar beta = 0.4f * dev_params.omega * dev_params.compactness *
  //               std::sin(theta) / (r * r);
  Scalar r1 = std::exp(dev_mesh.pos(0, n1 + 1, 0));
  Scalar r0 = std::exp(dev_mesh.pos(0, n1, 0));
  // Do the actual computation here
  // (Curl u)_1 = d2u3 - d3u2
  if (std::abs(dev_mesh.pos(1, n2, true) - CONST_PI) < 1.0e-5) {
    data.E1[globalOffset] +=
        dt *
        (-4.0f * (data.B3[globalOffset] - data.Bbg3[globalOffset]) *
             alpha_gr(r0) / (dev_mesh.delta[1] * r0) -
         // alpha_gr(r0) * j1[globalOffset]);
         data.J1[globalOffset]);
  } else {
    data.E1[globalOffset] +=
        // -dt * j1[globalOffset];
        dt * (((data.B3(n1, n2 + 1) - data.Bbg3(n1, n2 + 1)) *
                   alpha_gr(r0) * mesh_ptrs.l3_b(n1, n2 + 1) -
               (data.B3(n1, n2) - data.Bbg3(n1, n2)) * alpha_gr(r0) *
                   mesh_ptrs.l3_b(n1, n2)) /
                  mesh_ptrs.A1_e(n1, n2) -
              // alpha_gr(r0) * j1(n1, n2));
              data.J1(n1, n2));
  }
  // (Curl u)_2 = d3u1 - d1u3
  data.E2[globalOffset] +=
      // -dt * j2[globalOffset];
      dt * (((data.B3(n1, n2) - data.Bbg3(n1, n2)) * alpha_gr(r0) *
                 mesh_ptrs.l3_b(n1, n2) -
             (data.B3(n1 + 1, n2) - data.Bbg3(n1 + 1, n2)) *
                 alpha_gr(r1) * mesh_ptrs.l3_b(n1 + 1, n2)) /
                mesh_ptrs.A2_e(n1, n2) -
            // alpha_gr(r) * j2(n1, n2));
            data.J2(n1, n2));

  // (Curl u)_3 = d1u2 - d2u1
  data.E3[globalOffset] +=
      // -dt * j3[globalOffset];
      dt *
      (((data.B2(n1 + 1, n2) * alpha_gr(r1) *
             mesh_ptrs.l2_b(n1 + 1, n2) -
         data.Bbg2(n1 + 1, n2) * alpha_gr(r1) *
             mesh_ptrs.l2_b(n1 + 1, n2)) -
        (data.B2(n1, n2) * alpha_gr(r0) * mesh_ptrs.l2_b(n1, n2) -
         data.Bbg2(n1, n2) * alpha_gr(r0) * mesh_ptrs.l2_b(n1, n2)) +
        (data.B1(n1, n2) * alpha_gr(r) * mesh_ptrs.l1_b(n1, n2) -
         data.Bbg1(n1, n2) * alpha_gr(r) * mesh_ptrs.l1_b(n1, n2)) -
        (data.B1(n1, n2 + 1) * alpha_gr(r) *
             mesh_ptrs.l1_b(n1, n2 + 1) -
         data.Bbg1(n1, n2 + 1) * alpha_gr(r) *
             mesh_ptrs.l1_b(n1, n2 + 1))) /
           mesh_ptrs.A3_e(n1, n2) -
       // alpha_gr(r) * j3(n1, n2) + beta * rho);
       // j3(n1, n2) + beta * rho);
       data.J3(n1, n2));

  __syncthreads();
  // Extra work for the axis
  if (std::abs(dev_mesh.pos(1, n2, true) - dev_mesh.delta[1]) <
      1.0e-5) {
    n2 = dev_mesh.guard[1] - 1;
    globalOffset = data.E1.compute_offset(n1, n2);

    data.E3[globalOffset] = data.Ebg3[globalOffset];

    data.E1[globalOffset] +=
        dt * (4.0f * (data.B3(n1, n2 + 1) - data.Bbg3(n1, n2 + 1)) *
                  alpha_gr(r0) / (dev_mesh.delta[1] * r0) -
              // alpha_gr(r0) * j1[globalOffset]);
              data.J1[globalOffset]);
  }
}

// template <int DIM1, int DIM2>
__global__ void
compute_b_update_logsph(data_ptrs data, mesh_ptrs_log_sph mesh_ptrs,
                        Scalar dt) {
  int t1 = blockIdx.x, t2 = blockIdx.y;
  int c1 = threadIdx.x, c2 = threadIdx.y;
  int n1 = dev_mesh.guard[0] + t1 * blockDim.x + c1;
  int n2 = dev_mesh.guard[1] + t2 * blockDim.y + c2;
  // size_t globalOffset = n2 * e1.pitch + n1 * sizeof(Scalar);
  size_t globalOffset = data.E1.compute_offset(n1, n2);

  Scalar r1 = std::exp(dev_mesh.pos(0, n1, 1));
  Scalar r0 = std::exp(dev_mesh.pos(0, n1 - 1, 1));
  Scalar r = std::exp(dev_mesh.pos(0, n1, 0));
  // Do the actual computation here
  // (Curl u)_1 = d2u3 - d3u2
  data.B1[globalOffset] +=
      -dt *
      ((data.E3(n1, n2) - data.Ebg3(n1, n2)) * alpha_gr(r1) *
           mesh_ptrs.l3_e(n1, n2) -
       (data.E3(n1, n2 - 1) - data.Ebg3(n1, n2 - 1)) * alpha_gr(r1) *
           mesh_ptrs.l3_e(n1, n2 - 1)) /
      mesh_ptrs.A1_b(n1, n2);

  // (Curl u)_2 = d3u1 - d1u3
  data.B2[globalOffset] +=
      -dt *
      ((data.E3(n1 - 1, n2) - data.Ebg3(n1 - 1, n2)) * alpha_gr(r0) *
           mesh_ptrs.l3_e(n1 - 1, n2) -
       (data.E3(n1, n2) - data.Ebg3(n1, n2)) * alpha_gr(r1) *
           mesh_ptrs.l3_e(n1, n2)) /
      mesh_ptrs.A2_b(n1, n2);

  // (Curl u)_3 = d1u2 - d2u1
  data.B3[globalOffset] +=
      -dt *
      ((((data.E2(n1, n2) - data.Ebg2(n1, n2)) * alpha_gr(r1) +
         // (b1(n1, n2) + dev_bg_fields.B1(n1, n2)) * beta_phi(r1,
         // dev_mesh.pos(1, n2, 0))) *
         data.Bbg1(n1, n2) * beta_phi(r1, dev_mesh.pos(1, n2, 0))) *
            mesh_ptrs.l2_e(n1, n2) -
        ((data.E2(n1 - 1, n2) - data.Ebg2(n1 - 1, n2)) * alpha_gr(r0) +
         // (b1(n1 - 1, n2) + dev_bg_fields.B1(n1 - 1, n2)) *
         // beta_phi(r0, dev_mesh.pos(1, n2, 0))) *
         data.Bbg1(n1 - 1, n2) * beta_phi(r0, dev_mesh.pos(1, n2, 0))) *
            mesh_ptrs.l2_e(n1 - 1, n2) +
        ((data.E1(n1, n2 - 1) - data.Ebg1(n1, n2 - 1)) * alpha_gr(r) -
         // (b2(n1, n2 - 1) + dev_bg_fields.B2(n1, n2 - 1)) *
         // beta_phi(r, dev_mesh.pos(1, n2 - 1, 1))) *
         data.Bbg2(n1, n2 - 1) *
             beta_phi(r, dev_mesh.pos(1, n2 - 1, 1))) *
            mesh_ptrs.l1_e(n1, n2 - 1) -
        ((data.E1(n1, n2) - data.Ebg1(n1, n2)) * alpha_gr(r) -
         // (b2(n1, n2) + dev_bg_fields.B2(n1, n2)) * beta_phi(r,
         // dev_mesh.pos(1, n2, 1))) *
         data.Bbg2(n1, n2) * beta_phi(r, dev_mesh.pos(1, n2, 1))) *
            mesh_ptrs.l1_e(n1, n2)) /
       mesh_ptrs.A3_b(n1, n2));

  __syncthreads();

  // Extra work for the axis at theta = 0
  if (std::abs(dev_mesh.pos(1, n2, true) - dev_mesh.delta[1]) <
      1.0e-5) {
    n2 = dev_mesh.guard[1] - 1;
    globalOffset = data.B2.compute_offset(n1, n2);

    data.B2[globalOffset] = data.Bbg2[globalOffset];
  }
}

__global__ void
compute_double_circ_logsph(pitchptr<Scalar> b1, pitchptr<Scalar> b2,
                           pitchptr<Scalar> b3, pitchptr<Scalar> b01,
                           pitchptr<Scalar> b02, pitchptr<Scalar> b03,
                           pitchptr<Scalar> db1, pitchptr<Scalar> db2,
                           pitchptr<Scalar> db3,
                           mesh_ptrs_log_sph mesh_ptrs, Scalar coef) {
  int t1 = blockIdx.x, t2 = blockIdx.y;
  int c1 = threadIdx.x, c2 = threadIdx.y;
  int n1 = dev_mesh.guard[0] + t1 * blockDim.x + c1;
  int n2 = dev_mesh.guard[1] + t2 * blockDim.y + c2;

  size_t globalOffset = b1.compute_offset(n1, n2);

  db1[globalOffset] =
      b1[globalOffset] +
      coef *
          (mesh_ptrs.l3_e(n1, n2) *
               ((b1(n1, n2) - b01(n1, n2)) * mesh_ptrs.l1_b(n1, n2) -
                (b1(n1, n2 + 1) - b01(n1, n2 + 1)) *
                    mesh_ptrs.l1_b(n1, n2 + 1) +
                (b2(n1 + 1, n2) - b02(n1 + 1, n2)) *
                    mesh_ptrs.l2_b(n1 + 1, n2) -
                (b2(n1, n2) - b02(n1, n2)) * mesh_ptrs.l2_b(n1, n2)) /
               mesh_ptrs.A3_e(n1, n2) -
           mesh_ptrs.l3_e(n1, n2 - 1) *
               ((b1(n1, n2 - 1) - b01(n1, n2 - 1)) *
                    mesh_ptrs.l1_b(n1, n2 - 1) -
                (b1(n1, n2) - b01(n1, n2)) * mesh_ptrs.l1_b(n1, n2) +
                (b2(n1 + 1, n2 - 1) - b02(n1 + 1, n2 - 1)) *
                    mesh_ptrs.l2_b(n1 + 1, n2 - 1) -
                (b2(n1, n2 - 1) - b02(n1, n2 - 1)) *
                    mesh_ptrs.l2_b(n1, n2 - 1)) /
               mesh_ptrs.A3_e(n1, n2 - 1)) /
          mesh_ptrs.A1_b[globalOffset];

  db2[globalOffset] =
      b2[globalOffset] +
      coef *
          (mesh_ptrs.l3_e(n1 - 1, n2) *
               ((b1(n1 - 1, n2) - b01(n1 - 1, n2)) *
                    mesh_ptrs.l1_b(n1 - 1, n2) -
                (b1(n1 - 1, n2 + 1) - b01(n1 - 1, n2 + 1)) *
                    mesh_ptrs.l1_b(n1 - 1, n2 + 1) +
                (b2(n1, n2) - b02(n1, n2)) * mesh_ptrs.l2_b(n1, n2) -
                (b2(n1 - 1, n2) - b02(n1 - 1, n2)) *
                    mesh_ptrs.l2_b(n1 - 1, n2)) /
               mesh_ptrs.A3_e(n1 - 1, n2) -
           mesh_ptrs.l3_e(n1, n2) *
               ((b1(n1, n2) - b01(n1, n2)) * mesh_ptrs.l1_b(n1, n2) -
                (b1(n1, n2 + 1) - b01(n1, n2 + 1)) *
                    mesh_ptrs.l1_b(n1, n2 + 1) +
                (b2(n1 + 1, n2) - b02(n1 + 1, n2)) *
                    mesh_ptrs.l2_b(n1 + 1, n2) -
                (b2(n1, n2) - b02(n1, n2)) * mesh_ptrs.l2_b(n1, n2)) /
               mesh_ptrs.A3_e(n1, n2)) /
          mesh_ptrs.A2_b[globalOffset];

  db3[globalOffset] =
      b3[globalOffset] +
      coef *
          (mesh_ptrs.l1_e(n1, n2 - 1) *
               ((b3(n1, n2) - b03(n1, n2)) * mesh_ptrs.l3_b(n1, n2) -
                (b3(n1, n2 - 1) - b03(n1, n2 - 1)) *
                    mesh_ptrs.l3_b(n1, n2 - 1)) /
               mesh_ptrs.A3_e(n1, n2 - 1) -
           mesh_ptrs.l1_e(n1, n2) *
               ((b3(n1, n2 + 1) - b03(n1, n2 + 1)) *
                    mesh_ptrs.l3_b(n1, n2 + 1) -
                (b3(n1, n2) - b03(n1, n2)) * mesh_ptrs.l3_b(n1, n2)) /
               mesh_ptrs.A3_e(n1, n2) +
           mesh_ptrs.l2_e(n1, n2) *
               ((b3(n1, n2) - b03(n1, n2)) * mesh_ptrs.l3_b(n1, n2) -
                (b3(n1 + 1, n2) - b03(n1 + 1, n2)) *
                    mesh_ptrs.l3_b(n1 + 1, n2)) /
               mesh_ptrs.A3_e(n1, n2) -
           mesh_ptrs.l2_e(n1 - 1, n2) *
               ((b3(n1 - 1, n2) - b03(n1 - 1, n2)) *
                    mesh_ptrs.l3_b(n1 - 1, n2) -
                (b3(n1, n2) - b03(n1, n2)) * mesh_ptrs.l3_b(n1, n2)) /
               mesh_ptrs.A3_e(n1 - 1, n2)) /
          mesh_ptrs.A3_b[globalOffset];
}

__global__ void
compute_implicit_rhs(pitchptr<Scalar> db1, pitchptr<Scalar> db2,
                     pitchptr<Scalar> db3, pitchptr<Scalar> e1,
                     pitchptr<Scalar> e2, pitchptr<Scalar> e3,
                     pitchptr<Scalar> e01, pitchptr<Scalar> e02,
                     pitchptr<Scalar> e03, pitchptr<Scalar> j1,
                     pitchptr<Scalar> j2, pitchptr<Scalar> j3,
                     mesh_ptrs_log_sph mesh_ptrs, Scalar alpha,
                     Scalar beta, Scalar dt) {
  int t1 = blockIdx.x, t2 = blockIdx.y;
  int c1 = threadIdx.x, c2 = threadIdx.y;
  int n1 = dev_mesh.guard[0] + t1 * blockDim.x + c1;
  int n2 = dev_mesh.guard[1] + t2 * blockDim.y + c2;

  size_t globalOffset = db1.compute_offset(n1, n2);

  db1[globalOffset] +=
      -dt *
      ((alpha + beta) *
           ((e3(n1, n2) - e03(n1, n2)) * mesh_ptrs.l3_e(n1, n2) -
            (e3(n1, n2 - 1) - e03(n1, n2 - 1)) *
                mesh_ptrs.l3_e(n1, n2 - 1)) -
       dt * beta *
           (j3(n1, n2) * mesh_ptrs.l3_e(n1, n2) -
            j3(n1, n2 - 1) * mesh_ptrs.l3_e(n1, n2 - 1))) /
      mesh_ptrs.A1_b[globalOffset];

  // (Curl u)_2 = d3u1 - d1u3
  db2[globalOffset] +=
      -dt *
      ((alpha + beta) *
           ((e3(n1 - 1, n2) - e03(n1 - 1, n2)) *
                mesh_ptrs.l3_e(n1 - 1, n2) -
            (e3(n1, n2) - e03(n1, n2)) * mesh_ptrs.l3_e(n1, n2)) -
       dt * beta *
           (j3(n1 - 1, n2) * mesh_ptrs.l3_e(n1 - 1, n2) -
            j3(n1, n2) * mesh_ptrs.l3_e(n1, n2))) /
      mesh_ptrs.A2_b[globalOffset];

  // (Curl u)_3 = d1u2 - d2u1
  db3[globalOffset] +=
      -dt *
      ((alpha + beta) *
           ((e2(n1, n2) - e02(n1, n2)) * mesh_ptrs.l2_e(n1, n2) -
            (e2(n1 - 1, n2) - e02(n1 - 1, n2)) *
                mesh_ptrs.l2_e(n1 - 1, n2) +
            (e1(n1, n2 - 1) - e01(n1, n2 - 1)) *
                mesh_ptrs.l1_e(n1, n2 - 1) -
            (e1(n1, n2) - e01(n1, n2)) * mesh_ptrs.l1_e(n1, n2)) -
       dt * beta *
           (j2(n1, n2) * mesh_ptrs.l2_e(n1, n2) -
            j2(n1 - 1, n2) * mesh_ptrs.l2_e(n1 - 1, n2) +
            j1(n1, n2 - 1) * mesh_ptrs.l1_e(n1, n2 - 1) -
            j1(n1, n2) * mesh_ptrs.l1_e(n1, n2))) /
      mesh_ptrs.A3_b[globalOffset];
}

__global__ void
add_alpha_beta_B(pitchptr<Scalar> result, pitchptr<Scalar> b1, pitchptr<Scalar> b2,
                 size_t N1, size_t N2, Scalar alpha, Scalar beta) {
  for (int j = threadIdx.y + blockIdx.y * blockDim.y; j < N1; j += blockDim.y * gridDim.y) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < N1; i += blockDim.x * gridDim.x) {
      result(i, j) = alpha * b1(i, j) + beta * b2(i, j);
    }
  }
}

// template <int DIM1, int DIM2>
__global__ void
compute_divs_logsph(data_ptrs data, mesh_ptrs_log_sph mesh_ptrs) {
  int t1 = blockIdx.x, t2 = blockIdx.y;
  int c1 = threadIdx.x, c2 = threadIdx.y;
  int n1 = dev_mesh.guard[0] + t1 * blockDim.x + c1;
  int n2 = dev_mesh.guard[1] + t2 * blockDim.y + c2;
  // size_t globalOffset = n2 * divE.pitch + n1 * sizeof(Scalar);
  size_t globalOffset = data.divE.compute_offset(n1, n2);

  // if (n1 > dev_mesh.guard[0] + 1) {
  if (dev_mesh.pos(0, n1, 1) > dev_mesh.delta[0]) {
    data.divE[globalOffset] =
        ((data.E1(n1 + 1, n2) - data.Ebg1(n1 + 1, n2)) *
             mesh_ptrs.A1_e(n1 + 1, n2) -
         (data.E1(n1, n2) - data.Ebg1(n1, n2)) *
             mesh_ptrs.A1_e(n1, n2) +
         (data.E2(n1, n2 + 1) - data.Ebg2(n1, n2 + 1)) *
             mesh_ptrs.A2_e(n1, n2 + 1) -
         (data.E2(n1, n2) - data.Ebg2(n1, n2)) *
             mesh_ptrs.A2_e(n1, n2)) /
        (mesh_ptrs.dV(n1, n2) * dev_mesh.delta[0] * dev_mesh.delta[1]);

    // if (n2 == dev_mesh.dims[1] - dev_mesh.guard[1] - 1) {
    if (std::abs(dev_mesh.pos(1, n2, 1) - dev_mesh.sizes[1] +
                 dev_mesh.lower[1]) < 1.0e-5) {
      data.divE[globalOffset] =
          ((data.E1(n1 + 1, n2) - data.Ebg1(n1 + 1, n2)) *
               mesh_ptrs.A1_e(n1 + 1, n2) -
           (data.E1(n1, n2) - data.Ebg1(n1, n2)) *
               mesh_ptrs.A1_e(n1, n2) -
           // e2(n1, n2 + 1) *
           //     mesh_ptrs.A2_e(n1, n2 + 1) -
           2.0 * (data.E2(n1, n2) - data.Ebg2(n1, n2)) *
               mesh_ptrs.A2_e(n1, n2)) /
          (mesh_ptrs.dV(n1, n2) * dev_mesh.delta[0] *
           dev_mesh.delta[1]);
    }
  }
  data.divB[globalOffset] =
      ((data.B1(n1, n2) - data.Bbg1(n1, n2)) * mesh_ptrs.A1_b(n1, n2) -
       (data.B1(n1 - 1, n2) - data.Bbg1(n1 - 1, n2)) *
           mesh_ptrs.A1_b(n1 - 1, n2) +
       (data.B2(n1, n2) - data.Bbg2(n1, n2)) * mesh_ptrs.A2_b(n1, n2) -
       (data.B2(n1, n2 - 1) - data.Bbg2(n1, n2 - 1)) *
           mesh_ptrs.A2_b(n1, n2 - 1)) /
      (mesh_ptrs.dV(n1, n2) * dev_mesh.delta[0] * dev_mesh.delta[1]);

  __syncthreads();

  if (std::abs(dev_mesh.pos(1, n2, 1)) - dev_mesh.delta[1] < 1.0e-5) {
    n2 = dev_mesh.guard[1] - 1;
    globalOffset = data.E1.compute_offset(n1, n2);

    data.divE[globalOffset] =
        ((data.E1(n1 + 1, n2) - data.Ebg1(n1 + 1, n2)) *
             mesh_ptrs.A1_e(n1 + 1, n2) -
         (data.E1(n1, n2) - data.Ebg1(n1, n2)) *
             mesh_ptrs.A1_e(n1, n2) +
         2.0f * (data.E2(n1, n2 + 1) - data.Ebg2(n1, n2 + 1)) *
             mesh_ptrs.A2_e(n1, n2 + 1)) /
        (mesh_ptrs.dV(n1, n2) * dev_mesh.delta[0] * dev_mesh.delta[1]);
  }
}

__global__ void
stellar_boundary(data_ptrs data, Scalar omega) {
  for (int j = blockIdx.x * blockDim.x + threadIdx.x;
       j < dev_mesh.dims[1]; j += blockDim.x * gridDim.x) {
    Scalar theta_s = dev_mesh.pos(1, j, true);
    Scalar theta = dev_mesh.pos(1, j, false);
    // for (int i = 0; i < dev_mesh.guard[0] + 1; i++) {
    for (int i = 0; i < dev_mesh.guard[0]; i++) {
      Scalar r_s = std::exp(dev_mesh.pos(0, i, true));
      Scalar r = std::exp(dev_mesh.pos(0, i, false));

      Scalar coef = 0.0f;
      if (theta < 0.22f * CONST_PI && theta > 0.06f * CONST_PI)
        coef = 1.0f;
      else if (theta > 0.78f * CONST_PI && theta < 0.94f * CONST_PI)
        coef = -1.0f;

      data.B1(i, j) = data.Bbg1(i, j);
      data.B3(i, j) = data.Bbg3(i, j);
      data.E2(i, j) = (-omega * coef - 0.0 * dev_params.omega) *
                          std::sin(theta) * data.Bbg1(i, j) +
                      data.Ebg2(i, j);
      data.E1(i, j) = (omega * coef - 0.0 * dev_params.omega) *
                          std::sin(theta_s) * data.Bbg2(i, j) +
                      data.Ebg1(i, j);
      data.B2(i, j) = data.Bbg2(i, j);
      data.B3(i, j) = data.Bbg3(i, j);
    }
  }
}

__global__ void
axis_boundary_lower(data_ptrs data) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < dev_mesh.dims[0]; i += blockDim.x * gridDim.x) {
    data.E3(i, dev_mesh.guard[1] - 1) =
        data.Ebg3(i, dev_mesh.guard[1] - 1);
    // e3(i, dev_mesh.guard[1]) = 0.0f;
    data.E2(i, dev_mesh.guard[1] - 1) =
        -(data.E2(i, dev_mesh.guard[1]) -
          data.Ebg2(i, dev_mesh.guard[1])) +
        data.Ebg2(i, dev_mesh.guard[1] - 1);
    // e2(i, dev_mesh.guard[1] - 1) = e2(i, dev_mesh.guard[1]) = 0.0f;

    data.B3(i, dev_mesh.guard[1] - 1) =
        data.Bbg3(i, dev_mesh.guard[1] - 1);
    data.B3(i, dev_mesh.guard[1]) = data.Bbg3(i, dev_mesh.guard[1]);
    data.B2(i, dev_mesh.guard[1] - 1) =
        data.Bbg2(i, dev_mesh.guard[1] - 1);
    data.B1(i, dev_mesh.guard[1] - 1) =
        (data.B1(i, dev_mesh.guard[1]) -
         data.Bbg1(i, dev_mesh.guard[1])) +
        data.Bbg1(i, dev_mesh.guard[1] - 1);
  }
}

__global__ void
axis_boundary_upper(data_ptrs data) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < dev_mesh.dims[0]; i += blockDim.x * gridDim.x) {
    int j_last = dev_mesh.dims[1] - dev_mesh.guard[1];
    data.E3(i, j_last - 1) = data.Ebg3(i, j_last - 1);
    data.E2(i, j_last) =
        -(data.E2(i, j_last - 1) - data.Ebg2(i, j_last - 1)) +
        data.Ebg2(i, j_last);
    // e2(i, j_last) = e2(i, j_last - 1) = 0.0f;

    data.B3(i, j_last) = data.Bbg3(i, j_last);
    data.B3(i, j_last - 1) = data.Bbg3(i, j_last - 1);
    data.B2(i, j_last - 1) = data.Bbg2(i, j_last - 1);
    data.B1(i, j_last) =
        (data.B1(i, j_last - 1) - data.Bbg1(i, j_last - 1)) +
        data.Bbg1(i, j_last);
  }
}

__global__ void
outflow_boundary_sph(data_ptrs data) {
  for (int j = blockIdx.x * blockDim.x + threadIdx.x;
       j < dev_mesh.dims[1]; j += blockDim.x * gridDim.x) {
    for (int i = 0; i < dev_params.damping_length; i++) {
      int n1 = dev_mesh.dims[0] - dev_params.damping_length + i;
      // size_t offset = j * e1.pitch + n1 * sizeof(Scalar);
      size_t offset = data.E1.compute_offset(n1, j);
      Scalar lambda =
          1.0f - dev_params.damping_coef *
                     square((Scalar)i / dev_params.damping_length);
      data.E1[offset] = lambda * (data.E1[offset] - data.Ebg1[offset]) +
                        data.Ebg1[offset];
      data.E2[offset] = lambda * (data.E2[offset] - data.Ebg2[offset]) +
                        data.Ebg2[offset];
      data.E3[offset] = lambda * (data.E3[offset] - data.Ebg3[offset]) +
                        data.Ebg3[offset];
      // b1[offset] *= lambda;
      // b2[offset] *= lambda;
      data.B3[offset] = lambda * (data.B3[offset] - data.Bbg3[offset]) +
                        data.Bbg3[offset];
    }
  }
}

}  // namespace Kernels

field_solver_logsph::field_solver_logsph(sim_environment &env)
    : m_env(env) {
  m_tmp_e = multi_array<Scalar>(env.local_grid().extent());
  m_tmp_b1 = vector_field<Scalar>(env.local_grid());
  m_tmp_b2 = vector_field<Scalar>(env.local_grid());
}

field_solver_logsph::~field_solver_logsph() {}

void
field_solver_logsph::update_fields(sim_data &data, double dt,
                                   double time) {
  // Only implemented 2D!
  if (data.env.grid().dim() != 2) return;
  timer::stamp("field_update");

  // Assume E field guard cells are already in place

  Grid_LogSph &grid = *dynamic_cast<Grid_LogSph *>(&m_env.local_grid());
  auto mesh_ptrs = get_mesh_ptrs(grid);
  auto &mesh = grid.mesh();
  auto data_p = get_data_ptrs(data);

  dim3 blockSize(32, 16);
  dim3 gridSize(mesh.reduced_dim(0) / 32, mesh.reduced_dim(1) / 16);
  // Update B
  Kernels::compute_b_update_logsph<<<gridSize, blockSize>>>(
      data_p, mesh_ptrs, dt);
  CudaCheckError();

  CudaSafeCall(cudaDeviceSynchronize());
  // Communicate the new B values to guard cells
  m_env.send_guard_cells(data.B);
  // m_env.send_guard_cells(data.J);

  // Update E
  Kernels::compute_e_update_logsph<<<gridSize, blockSize>>>(
      data_p, mesh_ptrs, dt);
  CudaCheckError();

  CudaSafeCall(cudaDeviceSynchronize());

  // Communicate the new E values to guard cells
  m_env.send_guard_cells(data.E);

  // Compute divergences
  Kernels::compute_divs_logsph<<<gridSize, blockSize>>>(data_p,
                                                        mesh_ptrs);
  CudaCheckError();
  data.compute_edotb();
  m_env.send_guard_cells(data.EdotB);

  CudaSafeCall(cudaDeviceSynchronize());

  Logger::print_debug("e field smoothing {} times", 1);
  for (int i = 0; i < 1; i++) {
    filter_field(data.E, 0, grid);
    filter_field(data.E, 1, grid);
    filter_field(data.E, 2, grid);
    m_env.send_guard_cells(data.E);
  }

  timer::show_duration_since_stamp("Field update", "us",
                                   "field_update");
}

void
field_solver_logsph::update_fields_semi_impl(sim_data &data,
                                             double alpha, double beta,
                                             double dt, double time) {
  // Only implemented 2D!
  if (data.env.grid().dim() != 2) return;
  timer::stamp("field_update");

  // Assume all field guard cells are already in place

  // First compute (1 - alpha * beta * double_curl) B and store it in
  // m_tmp_b1
  compute_double_circ(data.B, data.Bbg, m_tmp_b1,
                      -alpha * beta * dt * dt);
  m_env.send_guard_cells(m_tmp_b1);

  // Then add the E and J terms to form the rhs
  compute_implicit_rhs(data, alpha, beta, m_tmp_b1, dt);

  // Define a double buffer to bounce around
  double_buffer<vector_field<Scalar>> buf(&m_tmp_b1, &m_tmp_b2);
  for (int i = 0; i < 5; i++) {
    // Use current to compute the double curl and store the result in
    // alternative
    compute_double_circ(*buf.current(), data.Bbg, *buf.alternative(),
                        -beta * beta * dt * dt);
    // Swap two buffers such that result is in current, and can start
    // next iteration
    buf.swap();
    m_env.send_guard_cells(*buf.current());
  }

  // Update E
  auto& mesh = m_env.grid().mesh();
  Kernels::add_alpha_beta_B<<<dim3(16, 16), dim3(32, 32)>>>
      (get_pitchptr(*buf.alternative(), 0), get_pitchptr(data.B, 0), get_pitchptr(*buf.current(), 0),
       mesh.dims[0], mesh.dims[1], alpha, beta);
  CudaCheckError();
  Kernels::add_alpha_beta_B<<<dim3(16, 16), dim3(32, 32)>>>
      (get_pitchptr(*buf.alternative(), 1), get_pitchptr(data.B, 1), get_pitchptr(*buf.current(), 1),
       mesh.dims[0], mesh.dims[1], alpha, beta);
  CudaCheckError();
  Kernels::add_alpha_beta_B<<<dim3(16, 16), dim3(32, 32)>>>
      (get_pitchptr(*buf.alternative(), 2), get_pitchptr(data.B, 2), get_pitchptr(*buf.current(), 2),
       mesh.dims[0], mesh.dims[1], alpha, beta);
  CudaCheckError();

  Grid_LogSph &grid = *dynamic_cast<Grid_LogSph *>(&m_env.local_grid());
  auto mesh_ptrs = get_mesh_ptrs(grid);
  auto data_p = get_data_ptrs(data);
  data_p.B1 = get_pitchptr(*buf.alternative(), 0);
  data_p.B2 = get_pitchptr(*buf.alternative(), 1);
  data_p.B3 = get_pitchptr(*buf.alternative(), 2);

  dim3 blockSize(32, 16);
  dim3 gridSize(mesh.reduced_dim(0) / 32, mesh.reduced_dim(1) / 16);
  Kernels::compute_e_update_logsph<<<gridSize, blockSize>>>(
      data_p, mesh_ptrs, dt);
  CudaCheckError();

  CudaSafeCall(cudaDeviceSynchronize());

  // Communicate the new E values to guard cells
  m_env.send_guard_cells(data.E);

  data.B.copy_from(*buf.current());

  // Compute divergences and edotb
  compute_divs(data);
  data.compute_edotb();
  m_env.send_guard_cells(data.EdotB);

  CudaSafeCall(cudaDeviceSynchronize());

  timer::show_duration_since_stamp("Field update", "us",
                                   "field_update");
}

// void
// field_solver_logsph::set_background_j(const vfield_t &J) {}

void
field_solver_logsph::apply_boundary(sim_data &data, double omega,
                                    double time) {
  auto data_p = get_data_ptrs(data);

  if (data.env.is_boundary(BoundaryPos::lower0)) {
    Kernels::stellar_boundary<<<32, 256>>>(data_p, omega);
    CudaCheckError();
  }

  if (data.env.is_boundary(BoundaryPos::upper0)) {
    Kernels::outflow_boundary_sph<<<32, 256>>>(data_p);
    CudaCheckError();
  }

  if (data.env.is_boundary(BoundaryPos::lower1)) {
    Kernels::axis_boundary_lower<<<32, 256>>>(data_p);
    CudaCheckError();
  }

  if (data.env.is_boundary(BoundaryPos::upper1)) {
    Kernels::axis_boundary_upper<<<32, 256>>>(data_p);
    CudaCheckError();
  }
  // Logger::print_info("omega is {}", omega);
}

void
field_solver_logsph::filter_field(vector_field<Scalar> &field, int comp,
                                  Grid_LogSph &grid) {
  auto mesh_ptrs = get_mesh_ptrs(grid);
  pitchptr<Scalar> A;
  if (comp == 0)
    A = mesh_ptrs.A1_e;
  else if (comp == 1)
    A = mesh_ptrs.A2_e;
  else if (comp == 2)
    A = mesh_ptrs.A3_e;

  auto &mesh = grid.mesh();
  dim3 blockSize(32, 16);
  dim3 gridSize(mesh.reduced_dim(0) / 32, mesh.reduced_dim(1) / 16);
  Kernels::filter_current_logsph<<<gridSize, blockSize>>>(
      get_pitchptr(field.data(comp)), get_pitchptr(m_tmp_e), A,
      m_env.is_boundary(0), m_env.is_boundary(1), m_env.is_boundary(2),
      m_env.is_boundary(3));
  field.data(comp).copy_from(m_tmp_e);
  CudaCheckError();
}

void
field_solver_logsph::compute_implicit_rhs(sim_data &data, double alpha,
                                          double beta,
                                          vector_field<Scalar> &result,
                                          double dt) {
  Grid_LogSph &grid = *dynamic_cast<Grid_LogSph *>(&m_env.local_grid());
  auto mesh_ptrs = get_mesh_ptrs(grid);
  auto &mesh = grid.mesh();
  dim3 blockSize(32, 16);
  dim3 gridSize(mesh.reduced_dim(0) / 32, mesh.reduced_dim(1) / 16);

  Kernels::compute_implicit_rhs<<<gridSize, blockSize>>>(
      get_pitchptr(result, 0), get_pitchptr(result, 1),
      get_pitchptr(result, 2), get_pitchptr(data.E, 0),
      get_pitchptr(data.E, 1), get_pitchptr(data.E, 2),
      get_pitchptr(data.Ebg, 0), get_pitchptr(data.Ebg, 1),
      get_pitchptr(data.Ebg, 2), get_pitchptr(data.J, 0),
      get_pitchptr(data.J, 1), get_pitchptr(data.J, 2), mesh_ptrs,
      alpha, beta, dt);
  CudaSafeCall(cudaDeviceSynchronize());
  CudaCheckError();
}

void
field_solver_logsph::compute_double_circ(vector_field<Scalar> &field,
                                         vector_field<Scalar> &field_bg,
                                         vector_field<Scalar> &result,
                                         double coef) {
  Grid_LogSph &grid = *dynamic_cast<Grid_LogSph *>(&m_env.local_grid());
  auto mesh_ptrs = get_mesh_ptrs(grid);
  auto &mesh = grid.mesh();
  dim3 blockSize(32, 16);
  dim3 gridSize(mesh.reduced_dim(0) / 32, mesh.reduced_dim(1) / 16);

  Kernels::compute_double_circ_logsph<<<gridSize, blockSize>>>(
      get_pitchptr(field, 0), get_pitchptr(field, 1),
      get_pitchptr(field, 2), get_pitchptr(field_bg, 0),
      get_pitchptr(field_bg, 1), get_pitchptr(field_bg, 2),
      get_pitchptr(result, 0), get_pitchptr(result, 1),
      get_pitchptr(result, 2), mesh_ptrs, coef);
  CudaSafeCall(cudaDeviceSynchronize());
  CudaCheckError();
}

void
field_solver_logsph::compute_divs(sim_data &data) {
  Grid_LogSph &grid = *dynamic_cast<Grid_LogSph *>(&m_env.local_grid());
  auto mesh_ptrs = get_mesh_ptrs(grid);
  auto &mesh = grid.mesh();
  auto data_ptr = get_data_ptrs(data);
  dim3 blockSize(32, 16);
  dim3 gridSize(mesh.reduced_dim(0) / 32, mesh.reduced_dim(1) / 16);

  Kernels::compute_divs_logsph<<<gridSize, blockSize>>>(data_ptr,
                                                        mesh_ptrs);
  CudaCheckError();
}

}  // namespace Aperture
