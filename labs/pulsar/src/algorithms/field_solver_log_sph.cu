#include "algorithms/field_solver_helper.cuh"
#include "algorithms/field_solver_log_sph.h"
#include "algorithms/finite_diff.h"
#include "cuda/constant_mem.h"
#include "cuda/cudaUtility.h"
#include "data/detail/multi_array_utils.hpp"
#include "data/field_data.h"
#include "data/fields_utils.h"
#include "utils/timer.h"

namespace Aperture {

namespace Kernels {

__device__ Scalar
beta_phi(Scalar r, Scalar theta) {
  return 0.4f * dev_params.compactness * dev_params.omega *
         std::sin(theta) / (r * r);
}

// template <int DIM1, int DIM2>
__global__ void
compute_e_update(cudaPitchedPtr e1, cudaPitchedPtr e2,
                 cudaPitchedPtr e3, cudaPitchedPtr b1,
                 cudaPitchedPtr b2, cudaPitchedPtr b3,
                 cudaPitchedPtr j1, cudaPitchedPtr j2,
                 cudaPitchedPtr j3, Grid_LogSph::mesh_ptrs mesh_ptrs,
                 Scalar dt) {
  // Load position parameters
  int t1 = blockIdx.x, t2 = blockIdx.y;
  int c1 = threadIdx.x, c2 = threadIdx.y;
  int n1 = dev_mesh.guard[0] + t1 * blockDim.x + c1;
  int n2 = dev_mesh.guard[1] + t2 * blockDim.y + c2;
  size_t globalOffset = n2 * e1.pitch + n1 * sizeof(Scalar);

  // Do the actual computation here
  // (Curl u)_1 = d2u3 - d3u2
  if (n2 == dev_mesh.dims[1] - dev_mesh.guard[1] - 1) {
    (*ptrAddr(e1, globalOffset)) +=
        dt *
        (-4.0f * *ptrAddr(b3, globalOffset) /
             (dev_mesh.delta[1] * std::exp(dev_mesh.pos(0, n1, 0))) -
         *ptrAddr(j1, globalOffset));
  } else {
    (*ptrAddr(e1, globalOffset)) +=
        // -dt * *ptrAddr(j1, globalOffset);
        dt * ((*ptrAddr(b3, globalOffset + b3.pitch) *
                   *ptrAddr(mesh_ptrs.l3_b, globalOffset + b3.pitch) -
               *ptrAddr(b3, globalOffset) *
                   *ptrAddr(mesh_ptrs.l3_b, globalOffset)) /
                  *ptrAddr(mesh_ptrs.A1_e, globalOffset) -
              *ptrAddr(j1, globalOffset));
  }
  // (Curl u)_2 = d3u1 - d1u3
  (*ptrAddr(e2, globalOffset)) +=
      // -dt * *ptrAddr(j2, globalOffset);
      dt *
      ((*ptrAddr(b3, globalOffset) *
            *ptrAddr(mesh_ptrs.l3_b, globalOffset) -
        *ptrAddr(b3, globalOffset + sizeof(Scalar)) *
            *ptrAddr(mesh_ptrs.l3_b, globalOffset + sizeof(Scalar))) /
           *ptrAddr(mesh_ptrs.A2_e, globalOffset) -
       *ptrAddr(j2, globalOffset));

  // (Curl u)_3 = d1u2 - d2u1
  (*ptrAddr(e3, globalOffset)) +=
      // -dt * *ptrAddr(j3, globalOffset);
      dt *
      ((*ptrAddr(b2, globalOffset + sizeof(Scalar)) *
            *ptrAddr(mesh_ptrs.l2_b, globalOffset + sizeof(Scalar)) -
        *ptrAddr(b2, globalOffset) *
            *ptrAddr(mesh_ptrs.l2_b, globalOffset) +
        *ptrAddr(b1, globalOffset) *
            *ptrAddr(mesh_ptrs.l1_b, globalOffset) -
        *ptrAddr(b1, globalOffset + b1.pitch) *
            *ptrAddr(mesh_ptrs.l1_b, globalOffset + b1.pitch)) /
           *ptrAddr(mesh_ptrs.A3_e, globalOffset) -
       *ptrAddr(j3, globalOffset));
  // (((*ptrAddr(b2, globalOffset + sizeof(Scalar)) -
  //    *ptrAddr(e1, globalOffset + sizeof(Scalar)) *
  //        beta_phi(std::exp(dev_mesh.pos(0, c1 + 1, 0)),
  //                 dev_mesh.pos(1, c2, 1))) *
  //       *ptrAddr(mesh_ptrs.l2_b, globalOffset + sizeof(Scalar)) -
  //   (*ptrAddr(b2, globalOffset) -
  //    *ptrAddr(e1, globalOffset) *
  //        beta_phi(std::exp(dev_mesh.pos(0, c1, 0)),
  //                 dev_mesh.pos(1, c2, 1))) *
  //       *ptrAddr(mesh_ptrs.l2_b, globalOffset) +
  //   (*ptrAddr(b1, globalOffset) +
  //    *ptrAddr(e2, globalOffset) *
  //        beta_phi(std::exp(dev_mesh.pos(0, c1, 1)),
  //                 dev_mesh.pos(1, c2, 0))) *
  //       *ptrAddr(mesh_ptrs.l1_b, globalOffset) -
  //   (*ptrAddr(b1, globalOffset + b1.pitch) +
  //    *ptrAddr(e2, globalOffset + e2.pitch) *
  //        beta_phi(std::exp(dev_mesh.pos(0, c1, 1)),
  //                 dev_mesh.pos(1, c2 + 1, 0))) *
  //       *ptrAddr(mesh_ptrs.l1_b, globalOffset + b1.pitch)) /
  //      *ptrAddr(mesh_ptrs.A3_e, globalOffset) -
  //  *ptrAddr(j3, globalOffset));

  __syncthreads();
  // Extra work for the axis
  // if (threadIdx.y == 0 && blockIdx.y == 0) {
  if (n2 == dev_mesh.guard[1]) {
    n2 = dev_mesh.guard[1] - 1;
    globalOffset = n2 * e1.pitch + n1 * sizeof(Scalar);

    // (*ptrAddr(e2, globalOffset)) = 0.0f;
    (*ptrAddr(e3, globalOffset)) = 0.0f;

    (*ptrAddr(e1, globalOffset)) +=
        dt *
        (4.0f * *ptrAddr(b3, globalOffset + b3.pitch) /
             (dev_mesh.delta[1] * std::exp(dev_mesh.pos(0, n1, 0))) -
         *ptrAddr(j1, globalOffset));
    // dt * ((*ptrAddr(b3, globalOffset + b3.pitch) *
    //            *ptrAddr(mesh_ptrs.l3_b, globalOffset + b3.pitch) -
    //        *ptrAddr(b3, globalOffset) *
    //            *ptrAddr(mesh_ptrs.l3_b, globalOffset)) /
    //           *ptrAddr(mesh_ptrs.A1_e, globalOffset) -
    //       *ptrAddr(j1, globalOffset));
    // if (n1 == 4) {
    //   printf("E1 is %f, %f\n", *ptrAddr(e1, globalOffset),
    //   *ptrAddr(e1, globalOffset + b3.pitch)); printf("E2 is %f,
    //   %f\n", *ptrAddr(e2, globalOffset), *ptrAddr(e2, globalOffset +
    //   b3.pitch)); printf("B3 is %f, %f\n", *ptrAddr(b3,
    //   globalOffset), *ptrAddr(b3, globalOffset + b3.pitch));
    // }
  }
}

// template <int DIM1, int DIM2>
__global__ void
compute_b_update(cudaPitchedPtr e1, cudaPitchedPtr e2,
                 cudaPitchedPtr e3, cudaPitchedPtr b1,
                 cudaPitchedPtr b2, cudaPitchedPtr b3,
                 Grid_LogSph::mesh_ptrs mesh_ptrs, Scalar dt) {
  int t1 = blockIdx.x, t2 = blockIdx.y;
  int c1 = threadIdx.x, c2 = threadIdx.y;
  int n1 = dev_mesh.guard[0] + t1 * blockDim.x + c1;
  int n2 = dev_mesh.guard[1] + t2 * blockDim.y + c2;
  size_t globalOffset = n2 * e1.pitch + n1 * sizeof(Scalar);

  // Do the actual computation here
  // (Curl u)_1 = d2u3 - d3u2
  (*ptrAddr(b1, globalOffset)) +=
      -dt *
      (*ptrAddr(e3, globalOffset) *
           *ptrAddr(mesh_ptrs.l3_e, globalOffset) -
       *ptrAddr(e3, globalOffset - e3.pitch) *
           *ptrAddr(mesh_ptrs.l3_e, globalOffset - e3.pitch)) /
      *ptrAddr(mesh_ptrs.A1_b, globalOffset);

  // (Curl u)_2 = d3u1 - d1u3
  (*ptrAddr(b2, globalOffset)) +=
      -dt *
      (*ptrAddr(e3, globalOffset - sizeof(Scalar)) *
           *ptrAddr(mesh_ptrs.l3_e, globalOffset - sizeof(Scalar)) -
       *ptrAddr(e3, globalOffset) *
           *ptrAddr(mesh_ptrs.l3_e, globalOffset)) /
      *ptrAddr(mesh_ptrs.A2_b, globalOffset);

  // (Curl u)_3 = d1u2 - d2u1
  (*ptrAddr(b3, globalOffset)) +=
      -dt *
      (((*ptrAddr(e2, globalOffset) +
         *ptrAddr(b1, globalOffset) *
             beta_phi(std::exp(dev_mesh.pos(0, c1, 1)),
                      dev_mesh.pos(1, c2, 0))) *
            *ptrAddr(mesh_ptrs.l2_e, globalOffset) -
        (*ptrAddr(e2, globalOffset - sizeof(Scalar)) +
         *ptrAddr(b1, globalOffset - sizeof(Scalar)) *
             beta_phi(std::exp(dev_mesh.pos(0, c1 - 1, 1)),
                      dev_mesh.pos(1, c2, 0))) *
            *ptrAddr(mesh_ptrs.l2_e, globalOffset - sizeof(Scalar)) +
        (*ptrAddr(e1, globalOffset - e1.pitch) -
         *ptrAddr(b2, globalOffset - b2.pitch) *
             beta_phi(std::exp(dev_mesh.pos(0, c1, 0)),
                      dev_mesh.pos(1, c2 - 1, 1))) *
            *ptrAddr(mesh_ptrs.l1_e, globalOffset - e1.pitch) -
        (*ptrAddr(e1, globalOffset) -
         *ptrAddr(b2, globalOffset) *
             beta_phi(std::exp(dev_mesh.pos(0, c1, 0)),
                      dev_mesh.pos(1, c2, 1))) *
            *ptrAddr(mesh_ptrs.l1_e, globalOffset)) /
       *ptrAddr(mesh_ptrs.A3_b, globalOffset));

  __syncthreads();

  // Extra work for the axis at theta = 0
  if (threadIdx.y == 0 && blockIdx.y == 0) {
    n2 = dev_mesh.guard[1] - 1;
    globalOffset = n2 * b1.pitch + n1 * sizeof(Scalar);

    // (*ptrAddr(b1, globalOffset)) +=
    //     -dt *
    //     (*ptrAddr(e3, globalOffset + e1.pitch) *
    //          *ptrAddr(mesh_ptrs.l3_e, globalOffset + e1.pitch) -
    //      *ptrAddr(e3, globalOffset) *
    //          *ptrAddr(mesh_ptrs.l3_e, globalOffset)) /
    //     *ptrAddr(mesh_ptrs.A1_b, globalOffset);
    // (*ptrAddr(b1, globalOffset)) +=
    //     -dt * *ptrAddr(e3, globalOffset + e3.pitch) *
    //     *ptrAddr(mesh_ptrs.l3_e, globalOffset + e3.pitch) /
    //     *ptrAddr(mesh_ptrs.A1_b, globalOffset);

    (*ptrAddr(b2, globalOffset)) = 0.0f;
  }
}

// template <int DIM1, int DIM2>
__global__ void
compute_divs(cudaPitchedPtr e1, cudaPitchedPtr e2, cudaPitchedPtr e3,
             cudaPitchedPtr b1, cudaPitchedPtr b2, cudaPitchedPtr b3,
             cudaPitchedPtr divE, cudaPitchedPtr divB,
             Grid_LogSph::mesh_ptrs mesh_ptrs) {
  int t1 = blockIdx.x, t2 = blockIdx.y;
  int c1 = threadIdx.x, c2 = threadIdx.y;
  int n1 = dev_mesh.guard[0] + t1 * blockDim.x + c1;
  int n2 = dev_mesh.guard[1] + t2 * blockDim.y + c2;
  size_t globalOffset = n2 * divE.pitch + n1 * sizeof(Scalar);

  if (n1 > dev_mesh.guard[0] + 1) {
    (*ptrAddr(divE, globalOffset)) =
        (*ptrAddr(e1, globalOffset + sizeof(Scalar)) *
             *ptrAddr(mesh_ptrs.A1_e, globalOffset + sizeof(Scalar)) -
         *ptrAddr(e1, globalOffset) *
             *ptrAddr(mesh_ptrs.A1_e, globalOffset) +
         *ptrAddr(e2, globalOffset + e2.pitch) *
             *ptrAddr(mesh_ptrs.A2_e, globalOffset + e2.pitch) -
         *ptrAddr(e2, globalOffset) *
             *ptrAddr(mesh_ptrs.A2_e, globalOffset)) /
        (*ptrAddr(mesh_ptrs.dV, globalOffset) * dev_mesh.delta[0] *
         dev_mesh.delta[1]);

    if (n2 == dev_mesh.dims[1] - dev_mesh.guard[1] - 1) {
      (*ptrAddr(divE, globalOffset)) =
          (*ptrAddr(e1, globalOffset + sizeof(Scalar)) *
               *ptrAddr(mesh_ptrs.A1_e, globalOffset + sizeof(Scalar)) -
           *ptrAddr(e1, globalOffset) *
               *ptrAddr(mesh_ptrs.A1_e, globalOffset) -
           // *ptrAddr(e2, globalOffset + e2.pitch) *
           //     *ptrAddr(mesh_ptrs.A2_e, globalOffset + e2.pitch) -
           2.0 * *ptrAddr(e2, globalOffset) *
               *ptrAddr(mesh_ptrs.A2_e, globalOffset)) /
          (*ptrAddr(mesh_ptrs.dV, globalOffset) * dev_mesh.delta[0] *
           dev_mesh.delta[1]);
    }
  }
  (*ptrAddr(divB, globalOffset)) =
      (*ptrAddr(b1, globalOffset) *
           *ptrAddr(mesh_ptrs.A1_b, globalOffset) -
       *ptrAddr(b1, globalOffset - sizeof(Scalar)) *
           *ptrAddr(mesh_ptrs.A1_b, globalOffset - sizeof(Scalar)) +
       *ptrAddr(b2, globalOffset) *
           *ptrAddr(mesh_ptrs.A2_b, globalOffset) -
       *ptrAddr(b2, globalOffset - b2.pitch) *
           *ptrAddr(mesh_ptrs.A2_b, globalOffset - b2.pitch)) /
      (*ptrAddr(mesh_ptrs.dV, globalOffset) * dev_mesh.delta[0] *
       dev_mesh.delta[1]);

  __syncthreads();
  if (n2 == dev_mesh.guard[1]) {
    n2 = dev_mesh.guard[1] - 1;
    globalOffset = n2 * e1.pitch + n1 * sizeof(Scalar);

    (*ptrAddr(divE, globalOffset)) =
        (*ptrAddr(e1, globalOffset + sizeof(Scalar)) *
             *ptrAddr(mesh_ptrs.A1_e, globalOffset + sizeof(Scalar)) -
         *ptrAddr(e1, globalOffset) *
             *ptrAddr(mesh_ptrs.A1_e, globalOffset) +
         2.0f * *ptrAddr(e2, globalOffset + e2.pitch) *
             *ptrAddr(mesh_ptrs.A2_e, globalOffset + e2.pitch)) /
        (*ptrAddr(mesh_ptrs.dV, globalOffset) * dev_mesh.delta[0] *
         dev_mesh.delta[1]);
  }
}

// template <int DIM2>
__global__ void
stellar_boundary(cudaPitchedPtr e1, cudaPitchedPtr e2,
                 cudaPitchedPtr e3, cudaPitchedPtr b1,
                 cudaPitchedPtr b2, cudaPitchedPtr b3, Scalar omega) {
  for (int j = blockIdx.x * blockDim.x + threadIdx.x;
       j < dev_mesh.dims[1]; j += blockDim.x * gridDim.x) {
    Scalar* row_e2 = ptrAddr(e2, j * e2.pitch);
    Scalar* row_b1 = ptrAddr(dev_bg_fields.B1, j * b1.pitch);
    Scalar* row_e1 = ptrAddr(e1, j * e1.pitch);
    Scalar* row_b2 = ptrAddr(dev_bg_fields.B2, j * b2.pitch);
    Scalar theta_s = dev_mesh.pos(1, j, true);
    Scalar theta = dev_mesh.pos(1, j, false);
    for (int i = 0; i < dev_mesh.guard[0] + 1; i++) {
      Scalar r_s = std::exp(dev_mesh.pos(0, i, true));
      Scalar r = std::exp(dev_mesh.pos(0, i, false));
      (*ptrAddr(b1, j * b1.pitch + i * sizeof(Scalar))) = 0.0f;
      (*ptrAddr(e3, j * e3.pitch + i * sizeof(Scalar))) = 0.0f;
      row_e2[i] = -omega * std::sin(theta) * r_s * row_b1[i];
      // Do not impose right on the surface
      row_e1[i] = omega * std::sin(theta_s) * r * row_b2[i];
      (*ptrAddr(b2, j * b2.pitch + i * sizeof(Scalar))) = 0.0f;
      (*ptrAddr(b3, j * b3.pitch + i * sizeof(Scalar))) = 0.0f;
    }
  }
}

__global__ void
axis_boundary(cudaPitchedPtr e1, cudaPitchedPtr e2, cudaPitchedPtr e3,
              cudaPitchedPtr b1, cudaPitchedPtr b2, cudaPitchedPtr b3) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < dev_mesh.dims[0]; i += blockDim.x * gridDim.x) {
    (*ptrAddr(e3, i, dev_mesh.guard[1] - 1)) = 0.0f;
    // (*ptrAddr(
    //     e1, (dev_mesh.guard[1] - 1) * e1.pitch + i * sizeof(Scalar)))
    //     = *ptrAddr(e1, dev_mesh.guard[1] * e1.pitch + i *
    //     sizeof(Scalar));
    (*ptrAddr(e2, i, dev_mesh.guard[1] - 1)) =
        -*ptrAddr(e2, i, dev_mesh.guard[1]);

    (*ptrAddr(b3, i, dev_mesh.guard[1] - 1)) =
        *ptrAddr(b3, i, dev_mesh.guard[1]);
    (*ptrAddr(b2, i, dev_mesh.guard[1] - 1)) = 0.0f;
    (*ptrAddr(b1, i, dev_mesh.guard[1] - 1)) =
        *ptrAddr(b1, i, dev_mesh.guard[1]);

    // (*ptrAddr(e3, dev_mesh.guard[1] * e3.pitch + i * sizeof(Scalar)))
    // =
    //     0.0f;

    // (*ptrAddr(e3,
    //           (dev_mesh.dims[1] - dev_mesh.guard[1] - 1) * e3.pitch +
    //               i * sizeof(Scalar))) = 0.0f;
    (*ptrAddr(e3, i, dev_mesh.dims[1] - dev_mesh.guard[1] - 1)) = 0.0f;
    (*ptrAddr(e2, i, dev_mesh.dims[1] - dev_mesh.guard[1])) =
        -*ptrAddr(e2, i, dev_mesh.dims[1] - dev_mesh.guard[1] - 1);

    (*ptrAddr(b3, i, dev_mesh.dims[1] - dev_mesh.guard[1])) =
        *ptrAddr(b3, i, dev_mesh.dims[1] - dev_mesh.guard[1] - 1);
    (*ptrAddr(b2, i, dev_mesh.dims[1] - dev_mesh.guard[1] - 1)) = 0.0f;
    (*ptrAddr(b1, i, dev_mesh.dims[1] - dev_mesh.guard[1])) =
        *ptrAddr(b1, i, dev_mesh.dims[1] - dev_mesh.guard[1] - 1);
    // (*ptrAddr(e1,
    //           (dev_mesh.dims[1] - dev_mesh.guard[1] - 1) * e1.pitch +
    //               i * sizeof(Scalar))) =
    //     *ptrAddr(e1,
    //              (dev_mesh.dims[1] - dev_mesh.guard[1] - 2) *
    //              e1.pitch +
    //                  i * sizeof(Scalar));
    // (*ptrAddr(b3,
    //           (dev_mesh.dims[1] - dev_mesh.guard[1] - 1) * b3.pitch +
    //               i * sizeof(Scalar))) = 0.0f;
    // (*ptrAddr(b1,
    //           (dev_mesh.dims[1] - dev_mesh.guard[1] - 1) * b1.pitch +
    //               i * sizeof(Scalar))) = 0.0f;
    // (*ptrAddr(b3, (dev_mesh.dims[1] - dev_mesh.guard[1]) * b3.pitch +
    //                   i * sizeof(Scalar))) =
    //     *ptrAddr(b3,
    //              (dev_mesh.dims[1] - dev_mesh.guard[1] - 1) *
    //              b3.pitch +
    //                  i * sizeof(Scalar));
    // (*ptrAddr(e2, (dev_mesh.dims[1] - dev_mesh.guard[1]) * e2.pitch +
    //                   i * sizeof(Scalar))) =
    //     -*ptrAddr(
    //         e2, (dev_mesh.dims[1] - dev_mesh.guard[1] - 1) * e2.pitch
    //         +
    //                 i * sizeof(Scalar));
  }
}

__global__ void
outflow_boundary(cudaPitchedPtr e1, cudaPitchedPtr e2,
                 cudaPitchedPtr e3, cudaPitchedPtr b1,
                 cudaPitchedPtr b2, cudaPitchedPtr b3) {
  for (int j = blockIdx.x * blockDim.x + threadIdx.x;
       j < dev_mesh.dims[1]; j += blockDim.x * gridDim.x) {
    for (int i = 0; i < dev_params.damping_length; i++) {
      int n1 = dev_mesh.dims[0] - dev_params.damping_length + i;
      size_t offset = j * e1.pitch + n1 * sizeof(Scalar);
      Scalar lambda =
          1.0f - dev_params.damping_coef *
                     square((Scalar)i / dev_params.damping_length);
      (*ptrAddr(e1, offset)) *= lambda;
      (*ptrAddr(e2, offset)) *= lambda;
      (*ptrAddr(e3, offset)) *= lambda;
      // (*ptrAddr(b1, offset)) *= lambda;
      // (*ptrAddr(b2, offset)) *= lambda;
      (*ptrAddr(b3, offset)) *= lambda;
    }
  }
}

__global__ void
relax_electric_potential(cudaPitchedPtr e1, cudaPitchedPtr e2,
                         cudaPitchedPtr* rho, cudaPitchedPtr dphi,
                         Grid_LogSph::mesh_ptrs mesh_ptrs) {
  int t1 = blockIdx.x, t2 = blockIdx.y;
  int c1 = threadIdx.x, c2 = threadIdx.y;
  int n1 = dev_mesh.guard[0] + t1 * blockDim.x + c1;
  int n2 = dev_mesh.guard[1] + t2 * blockDim.y + c2;
  size_t globalOffset = n2 * e1.pitch + n1 * sizeof(Scalar);

  // (*ptrAddr(dphi, globalOffset));
  Scalar rho_total = 0.0f;
  for (int n = 0; n < dev_params.num_species; n++) {
    rho_total += *ptrAddr(rho[n], globalOffset);
  }
  Scalar diff = *ptrAddr(mesh_ptrs.dV, globalOffset) * rho_total;
  // dev_mesh.delta[0] * dev_mesh.delta[1] * rho_total;
  if (n1 > dev_mesh.guard[0] + 1) {
    diff -=
        (*ptrAddr(e1, globalOffset + sizeof(Scalar)) *
             *ptrAddr(mesh_ptrs.A1_e, globalOffset + sizeof(Scalar)) -
         *ptrAddr(e1, globalOffset) *
             *ptrAddr(mesh_ptrs.A1_e, globalOffset) +
         *ptrAddr(e2, globalOffset + e2.pitch) *
             *ptrAddr(mesh_ptrs.A2_e, globalOffset + e2.pitch) -
         *ptrAddr(e2, globalOffset) *
             *ptrAddr(mesh_ptrs.A2_e, globalOffset));
  }
  Scalar r0 = std::exp(dev_mesh.pos(0, n1, false));
  Scalar r1s = std::exp(dev_mesh.pos(0, n1 + 1, false));
  Scalar r1 = std::exp(dev_mesh.pos(0, n1, true));

  Scalar Ar1 =
      *ptrAddr(mesh_ptrs.A1_e, globalOffset + sizeof(Scalar)) / r1s;
  Scalar Ar0 = *ptrAddr(mesh_ptrs.A1_e, globalOffset) / r0;
  Scalar At1 = *ptrAddr(mesh_ptrs.A2_e, globalOffset + e2.pitch) / r1;
  Scalar At0 = *ptrAddr(mesh_ptrs.A2_e, globalOffset) / r1;

  Scalar Atot = (Ar1 + Ar0 + At1 + At0);

  if (n1 > dev_mesh.guard[0] + 1) {
    (*ptrAddr(dphi, globalOffset)) =
        (diff + (Ar1 * *ptrAddr(dphi, globalOffset + sizeof(Scalar)) +
                 Ar0 * *ptrAddr(dphi, globalOffset - sizeof(Scalar)) +
                 At1 * *ptrAddr(dphi, globalOffset + dphi.pitch) +
                 At0 * *ptrAddr(dphi, globalOffset - dphi.pitch))) /
        Atot;
  }
}

__global__ void
correct_E_field(cudaPitchedPtr e1, cudaPitchedPtr e2,
                cudaPitchedPtr dphi, Grid_LogSph::mesh_ptrs mesh_ptrs) {
  int t1 = blockIdx.x, t2 = blockIdx.y;
  int c1 = threadIdx.x, c2 = threadIdx.y;
  int n1 = dev_mesh.guard[0] + t1 * blockDim.x + c1;
  int n2 = dev_mesh.guard[1] + t2 * blockDim.y + c2;
  size_t globalOffset = n2 * e1.pitch + n1 * sizeof(Scalar);

  if (n1 > dev_mesh.guard[0] + 1) {
    (*ptrAddr(e1, globalOffset)) -=
        (*ptrAddr(dphi, globalOffset) -
         *ptrAddr(dphi, globalOffset - sizeof(Scalar))) /
        (std::exp(dev_mesh.pos(0, n1, false)) * dev_mesh.delta[0]);
    (*ptrAddr(e2, globalOffset)) -=
        (*ptrAddr(dphi, globalOffset) -
         *ptrAddr(dphi, globalOffset - dphi.pitch)) /
        (std::exp(dev_mesh.pos(0, n1, true)) * dev_mesh.delta[1]);

    if (blockIdx.y == 0 && threadIdx.y == 0) {
      n2 = dev_mesh.guard[1] - 1;
      globalOffset = n2 * e1.pitch + n1 * sizeof(Scalar);

      (*ptrAddr(e1, globalOffset)) -=
          (*ptrAddr(dphi, globalOffset) -
           *ptrAddr(dphi, globalOffset - sizeof(Scalar))) /
          (std::exp(dev_mesh.pos(0, n1, false)) * dev_mesh.delta[0]);
    }
  }
}

}  // namespace Kernels

FieldSolver_LogSph::FieldSolver_LogSph(const Grid_LogSph& g)
    : m_grid(g), m_divE(g), m_divB(g), m_phi_e(g) {
  m_divB.set_stagger(0b000);
}

FieldSolver_LogSph::~FieldSolver_LogSph() {
  if (m_dev_rho != nullptr) {
    CudaSafeCall(cudaFree(m_dev_rho));
  }
}

void
FieldSolver_LogSph::update_fields(SimData& data, double dt,
                                  double time) {
  update_fields(data.E, data.B, data.J, dt, time);
}

void
FieldSolver_LogSph::update_fields(vfield_t& E, vfield_t& B,
                                  const vfield_t& J, double dt,
                                  double time) {
  Logger::print_info("Updating fields");
  auto mesh_ptrs = m_grid.get_mesh_ptrs();
  auto& mesh = m_grid.mesh();

  if (m_grid.dim() == 2) {
    // We only implemented 2d at the moment
    dim3 blockSize(32, 16);
    dim3 gridSize(mesh.reduced_dim(0) / 32, mesh.reduced_dim(1) / 16);
    // Update B
    Kernels::compute_b_update<<<gridSize, blockSize>>>(
        E.ptr(0), E.ptr(1), E.ptr(2), B.ptr(0), B.ptr(1), B.ptr(2),
        mesh_ptrs, dt);
    CudaCheckError();
    cudaDeviceSynchronize();

    // Update E
    Kernels::compute_e_update<<<gridSize, blockSize>>>(
        E.ptr(0), E.ptr(1), E.ptr(2), B.ptr(0), B.ptr(1), B.ptr(2),
        J.ptr(0), J.ptr(1), J.ptr(2), mesh_ptrs, dt);
    CudaCheckError();
    // cudaDeviceSynchronize();

    // if (m_comm_callback_vfield != nullptr) {
    //   m_comm_callback_vfield(E);
    //   m_comm_callback_vfield(B);
    // }

    // Compute divE
    Kernels::compute_divs<<<gridSize, blockSize>>>(
        E.ptr(0), E.ptr(1), E.ptr(2), B.ptr(0), B.ptr(1), B.ptr(2),
        m_divE.ptr(), m_divB.ptr(), mesh_ptrs);
    CudaCheckError();
  }
}

void
FieldSolver_LogSph::set_background_j(const vfield_t& J) {}

void
FieldSolver_LogSph::boundary_conditions(SimData& data, double omega) {
  // Logger::print_info("omega is {}", omega);
  Kernels::stellar_boundary<<<32, 256>>>(
      data.E.ptr(0), data.E.ptr(1), data.E.ptr(2), data.B.ptr(0),
      data.B.ptr(1), data.B.ptr(2), omega);
  CudaCheckError();

  Kernels::axis_boundary<<<32, 256>>>(data.E.ptr(0), data.E.ptr(1),
                                      data.E.ptr(2), data.B.ptr(0),
                                      data.B.ptr(1), data.B.ptr(2));
  CudaCheckError();

  Kernels::outflow_boundary<<<32, 256>>>(data.E.ptr(0), data.E.ptr(1),
                                         data.E.ptr(2), data.B.ptr(0),
                                         data.B.ptr(1), data.B.ptr(2));
  CudaCheckError();
}

void
FieldSolver_LogSph::clean_divergence(SimData& data) {
  init_dev_rho(data);
  m_phi_e.initialize();

  auto& mesh = m_grid.mesh();
  auto mesh_ptrs = m_grid.get_mesh_ptrs();

  dim3 blockSize(32, 16);
  dim3 gridSize(mesh.reduced_dim(0) / 32, mesh.reduced_dim(1) / 16);

  for (int n = 0; n < 1; n++) {
    Kernels::relax_electric_potential<<<gridSize, blockSize>>>(
        data.E.ptr(0), data.E.ptr(1), m_dev_rho, m_phi_e.ptr(),
        mesh_ptrs);
    CudaCheckError();

    Kernels::correct_E_field<<<gridSize, blockSize>>>(
        data.E.ptr(0), data.E.ptr(1), m_phi_e.ptr(), mesh_ptrs);
    CudaCheckError();
  }
}

void
FieldSolver_LogSph::init_dev_rho(SimData& data) {
  if (!m_rho_initialized) {
    if (m_dev_rho == nullptr) {
      CudaSafeCall(cudaMallocManaged(
          &m_dev_rho, data.num_species * sizeof(cudaPitchedPtr)));
    }
    for (int i = 0; i < data.num_species; i++) {
      m_dev_rho[i] = data.Rho[i].ptr();
    }
    m_rho_initialized = true;
  }
}

}  // namespace Aperture