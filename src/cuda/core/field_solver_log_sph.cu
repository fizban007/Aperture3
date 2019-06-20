#include "core/detail/multi_array_utils.hpp"
#include "cuda/constant_mem.h"
#include "cuda/core/field_solver_helper.cuh"
#include "cuda/core/field_solver_log_sph.h"
#include "cuda/core/finite_diff.h"
#include "cuda/cudaUtility.h"
#include "cuda/data/field_data.h"
#include "cuda/data/fields_utils.h"
#include "cuda/ptr_util.h"
#include "cuda/utils/iterate_devices.h"
#include "cuda/utils/pitchptr.cuh"
#include "utils/timer.h"

namespace Aperture {

namespace Kernels {

__device__ Scalar
beta_phi(Scalar r, Scalar theta) {
  return -0.4f * dev_params.compactness * dev_params.omega *
         std::sin(theta) / (r * r);
}

__device__ Scalar
alpha_gr(Scalar r) {
  // return std::sqrt(1.0f - dev_params.compactness / r);
  return 1.0f;
}

// template <int DIM1, int DIM2>
__global__ void
compute_e_update(pitchptr<Scalar> e1, pitchptr<Scalar> e2,
                 pitchptr<Scalar> e3, pitchptr<Scalar> b1,
                 pitchptr<Scalar> b2, pitchptr<Scalar> b3,
                 pitchptr<Scalar> j1, pitchptr<Scalar> j2,
                 pitchptr<Scalar> j3, pitchptr<Scalar> rho0,
                 pitchptr<Scalar> rho1, pitchptr<Scalar> rho2,
                 Grid_LogSph_dev::mesh_ptrs mesh_ptrs, Scalar dt) {
  // Load position parameters
  int t1 = blockIdx.x, t2 = blockIdx.y;
  int c1 = threadIdx.x, c2 = threadIdx.y;
  int n1 = dev_mesh.guard[0] + t1 * blockDim.x + c1;
  int n2 = dev_mesh.guard[1] + t2 * blockDim.y + c2;
  size_t globalOffset = e1.compute_offset(n1, n2);

  Scalar r = std::exp(dev_mesh.pos(0, n1, true));
  Scalar theta = dev_mesh.pos(1, n2, true);
  Scalar theta0 = dev_mesh.pos(1, n2, false);
  // Scalar beta = 0.4f * dev_params.omega * dev_params.compactness *
  //               std::sin(theta) / (r * r);
  Scalar rho =
      rho0[globalOffset] + rho1[globalOffset] + rho2[globalOffset];
  Scalar r1 = std::exp(dev_mesh.pos(0, n1 + 1, 0));
  Scalar r0 = std::exp(dev_mesh.pos(0, n1, 0));
  // Do the actual computation here
  // (Curl u)_1 = d2u3 - d3u2
  if (std::abs(dev_mesh.pos(1, n2, true) - CONST_PI) < 1.0e-5) {
    e1[globalOffset] += dt * (-4.0f * b3[globalOffset] * alpha_gr(r0) /
                                  (dev_mesh.delta[1] * r0) -
                              // alpha_gr(r0) * j1[globalOffset]);
                              j1[globalOffset]);
  } else {
    e1[globalOffset] +=
        // -dt * j1[globalOffset];
        dt *
        ((b3(n1, n2 + 1) * alpha_gr(r0) * mesh_ptrs.l3_b(n1, n2 + 1) -
          b3(n1, n2) * alpha_gr(r0) * mesh_ptrs.l3_b(n1, n2)) /
             mesh_ptrs.A1_e(n1, n2) -
         // alpha_gr(r0) * j1(n1, n2));
         j1(n1, n2));
  }
  // (Curl u)_2 = d3u1 - d1u3
  e2[globalOffset] +=
      // -dt * j2[globalOffset];
      dt *
      ((b3(n1, n2) * alpha_gr(r0) * mesh_ptrs.l3_b(n1, n2) -
        b3(n1 + 1, n2) * alpha_gr(r1) * mesh_ptrs.l3_b(n1 + 1, n2)) /
           mesh_ptrs.A2_e(n1, n2) -
       // alpha_gr(r) * j2(n1, n2));
       j2(n1, n2));

  // (Curl u)_3 = d1u2 - d2u1
  e3[globalOffset] +=
      // -dt * j3[globalOffset];
      dt * ((b2(n1 + 1, n2) * alpha_gr(r1) *
                 // e1(n1 + 1, n2) * beta_phi(r1, theta)) *
                 mesh_ptrs.l2_b(n1 + 1, n2) -
             b2(n1, n2) * alpha_gr(r0) *
                 // - e1(n1, n2) * beta_phi(r0, theta)) *
                 mesh_ptrs.l2_b(n1, n2) +
             b1(n1, n2) * alpha_gr(r) *
                 // + e2(n1, n2) * beta_phi(r, theta0)) *
                 mesh_ptrs.l1_b(n1, n2) -
             b1(n1, n2 + 1) * alpha_gr(r) *
                 // e2(n1, n2 + 1) * beta_phi(r, theta0 +
                 // dev_mesh.delta[1])) *
                 mesh_ptrs.l1_b(n1, n2 + 1)) /
                mesh_ptrs.A3_e(n1, n2) -
            // alpha_gr(r) * j3(n1, n2) + beta * rho);
            // j3(n1, n2) + beta * rho);
            j3(n1, n2));

  __syncthreads();
  // Extra work for the axis
  if (std::abs(dev_mesh.pos(1, n2, true) - dev_mesh.delta[1]) <
      1.0e-5) {
    n2 = dev_mesh.guard[1] - 1;
    globalOffset = e1.compute_offset(n1, n2);

    e3[globalOffset] = 0.0f;

    e1[globalOffset] += dt * (4.0f * b3(n1, n2 + 1) * alpha_gr(r0) /
                                  (dev_mesh.delta[1] * r0) -
                              // alpha_gr(r0) * j1[globalOffset]);
                              j1[globalOffset]);
  }
}

// template <int DIM1, int DIM2>
__global__ void
compute_b_update(pitchptr<Scalar> e1, pitchptr<Scalar> e2,
                 pitchptr<Scalar> e3, pitchptr<Scalar> b1,
                 pitchptr<Scalar> b2, pitchptr<Scalar> b3,
                 Grid_LogSph_dev::mesh_ptrs mesh_ptrs, Scalar dt) {
  int t1 = blockIdx.x, t2 = blockIdx.y;
  int c1 = threadIdx.x, c2 = threadIdx.y;
  int n1 = dev_mesh.guard[0] + t1 * blockDim.x + c1;
  int n2 = dev_mesh.guard[1] + t2 * blockDim.y + c2;
  // size_t globalOffset = n2 * e1.pitch + n1 * sizeof(Scalar);
  size_t globalOffset = e1.compute_offset(n1, n2);

  Scalar r1 = std::exp(dev_mesh.pos(0, n1, 1));
  Scalar r0 = std::exp(dev_mesh.pos(0, n1 - 1, 1));
  Scalar r = std::exp(dev_mesh.pos(0, n1, 0));
  // Do the actual computation here
  // (Curl u)_1 = d2u3 - d3u2
  b1[globalOffset] +=
      -dt *
      (e3(n1, n2) * alpha_gr(r1) * mesh_ptrs.l3_e(n1, n2) -
       e3(n1, n2 - 1) * alpha_gr(r1) * mesh_ptrs.l3_e(n1, n2 - 1)) /
      mesh_ptrs.A1_b(n1, n2);

  // (Curl u)_2 = d3u1 - d1u3
  b2[globalOffset] +=
      -dt *
      (e3(n1 - 1, n2) * alpha_gr(r0) * mesh_ptrs.l3_e(n1 - 1, n2) -
       e3(n1, n2) * alpha_gr(r1) * mesh_ptrs.l3_e(n1, n2)) /
      mesh_ptrs.A2_b(n1, n2);

  // (Curl u)_3 = d1u2 - d2u1
  b3[globalOffset] +=
      -dt *
      (((e2(n1, n2) * alpha_gr(r1) +
         // (b1(n1, n2) + dev_bg_fields.B1(n1, n2)) * beta_phi(r1, dev_mesh.pos(1, n2, 0))) *
         dev_bg_fields.B1(n1, n2) * beta_phi(r1, dev_mesh.pos(1, n2, 0))) *
            mesh_ptrs.l2_e(n1, n2) -
        (e2(n1 - 1, n2) * alpha_gr(r0) +
         // (b1(n1 - 1, n2) + dev_bg_fields.B1(n1 - 1, n2)) * beta_phi(r0, dev_mesh.pos(1, n2, 0))) *
         dev_bg_fields.B1(n1 - 1, n2) * beta_phi(r0, dev_mesh.pos(1, n2, 0))) *
            mesh_ptrs.l2_e(n1 - 1, n2) +
        (e1(n1, n2 - 1) * alpha_gr(r) -
         // (b2(n1, n2 - 1) + dev_bg_fields.B2(n1, n2 - 1)) * beta_phi(r, dev_mesh.pos(1, n2 - 1, 1))) *
         dev_bg_fields.B2(n1, n2 - 1) * beta_phi(r, dev_mesh.pos(1, n2 - 1, 1))) *
            mesh_ptrs.l1_e(n1, n2 - 1) -
        (e1(n1, n2) * alpha_gr(r) -
         // (b2(n1, n2) + dev_bg_fields.B2(n1, n2)) * beta_phi(r, dev_mesh.pos(1, n2, 1))) *
         dev_bg_fields.B2(n1, n2) * beta_phi(r, dev_mesh.pos(1, n2, 1))) *
            mesh_ptrs.l1_e(n1, n2)) /
       mesh_ptrs.A3_b(n1, n2));

  __syncthreads();

  // Extra work for the axis at theta = 0
  if (std::abs(dev_mesh.pos(1, n2, true) - dev_mesh.delta[1]) <
      1.0e-5) {
    n2 = dev_mesh.guard[1] - 1;
    globalOffset = b2.compute_offset(n1, n2);

    b2[globalOffset] = 0.0f;
  }
}

// template <int DIM1, int DIM2>
__global__ void
compute_divs(pitchptr<Scalar> e1, pitchptr<Scalar> e2,
             pitchptr<Scalar> e3, pitchptr<Scalar> b1,
             pitchptr<Scalar> b2, pitchptr<Scalar> b3,
             pitchptr<Scalar> divE, pitchptr<Scalar> divB,
             Grid_LogSph_dev::mesh_ptrs mesh_ptrs) {
  int t1 = blockIdx.x, t2 = blockIdx.y;
  int c1 = threadIdx.x, c2 = threadIdx.y;
  int n1 = dev_mesh.guard[0] + t1 * blockDim.x + c1;
  int n2 = dev_mesh.guard[1] + t2 * blockDim.y + c2;
  // size_t globalOffset = n2 * divE.pitch + n1 * sizeof(Scalar);
  size_t globalOffset = divE.compute_offset(n1, n2);

  // if (n1 > dev_mesh.guard[0] + 1) {
  if (dev_mesh.pos(0, n1, 1) > dev_mesh.delta[0]) {
    divE[globalOffset] =
        (e1(n1 + 1, n2) * mesh_ptrs.A1_e(n1 + 1, n2) -
         e1(n1, n2) * mesh_ptrs.A1_e(n1, n2) +
         e2(n1, n2 + 1) * mesh_ptrs.A2_e(n1, n2 + 1) -
         e2(n1, n2) * mesh_ptrs.A2_e(n1, n2)) /
        (mesh_ptrs.dV(n1, n2) * dev_mesh.delta[0] * dev_mesh.delta[1]);

    // if (n2 == dev_mesh.dims[1] - dev_mesh.guard[1] - 1) {
    if (std::abs(dev_mesh.pos(1, n2, 1) - dev_mesh.sizes[1] +
                 dev_mesh.lower[1]) < 1.0e-5) {
      divE[globalOffset] =
          (e1(n1 + 1, n2) * mesh_ptrs.A1_e(n1 + 1, n2) -
           e1(n1, n2) * mesh_ptrs.A1_e(n1, n2) -
           // e2(n1, n2 + 1) *
           //     mesh_ptrs.A2_e(n1, n2 + 1) -
           2.0 * e2(n1, n2) * mesh_ptrs.A2_e(n1, n2)) /
          (mesh_ptrs.dV(n1, n2) * dev_mesh.delta[0] *
           dev_mesh.delta[1]);
    }
  }
  divB[globalOffset] =
      (b1(n1, n2) * mesh_ptrs.A1_b(n1, n2) -
       b1(n1 - 1, n2) * mesh_ptrs.A1_b(n1 - 1, n2) +
       b2(n1, n2) * mesh_ptrs.A2_b(n1, n2) -
       b2(n1, n2 - 1) * mesh_ptrs.A2_b(n1, n2 - 1)) /
      (mesh_ptrs.dV(n1, n2) * dev_mesh.delta[0] * dev_mesh.delta[1]);

  __syncthreads();

  if (std::abs(dev_mesh.pos(1, n2, 1)) - dev_mesh.delta[1] < 1.0e-5) {
    n2 = dev_mesh.guard[1] - 1;
    globalOffset = e1.compute_offset(n1, n2);

    divE[globalOffset] =
        (e1(n1 + 1, n2) * mesh_ptrs.A1_e(n1 + 1, n2) -
         e1(n1, n2) * mesh_ptrs.A1_e(n1, n2) +
         2.0f * e2(n1, n2 + 1) * mesh_ptrs.A2_e(n1, n2 + 1)) /
        (mesh_ptrs.dV(n1, n2) * dev_mesh.delta[0] * dev_mesh.delta[1]);
  }
}

__global__ void
stellar_boundary(pitchptr<Scalar> e1, pitchptr<Scalar> e2,
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
                 dev_bg_fields.B1(i, j) / alpha_gr(r_s) / r_s / r_s;
      e1(i, j) = (omega - omega_LT) * std::sin(theta_s) *
                 dev_bg_fields.B2(i, j) / alpha_gr(r_s) / r / r;
      b2(i, j) = 0.0f;
      b3(i, j) = 0.0f;
    }
  }
}

__global__ void
axis_boundary_lower(pitchptr<Scalar> e1, pitchptr<Scalar> e2,
                    pitchptr<Scalar> e3, pitchptr<Scalar> b1,
                    pitchptr<Scalar> b2, pitchptr<Scalar> b3) {
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
axis_boundary_upper(pitchptr<Scalar> e1, pitchptr<Scalar> e2,
                    pitchptr<Scalar> e3, pitchptr<Scalar> b1,
                    pitchptr<Scalar> b2, pitchptr<Scalar> b3) {
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
outflow_boundary(pitchptr<Scalar> e1, pitchptr<Scalar> e2,
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

// __global__ void
// relax_electric_potential(pitchptr<Scalar e1,
// pitchptr<Scalar e2,
//                          pitchptr<Scalar* rho,
//                          pitchptr<Scalar dphi,
//                          Grid_LogSph_dev::mesh_ptrs mesh_ptrs) {
//   int t1 = blockIdx.x, t2 = blockIdx.y;
//   int c1 = threadIdx.x, c2 = threadIdx.y;
//   int n1 = dev_mesh.guard[0] + t1 * blockDim.x + c1;
//   int n2 = dev_mesh.guard[1] + t2 * blockDim.y + c2;
//   size_t globalOffset = n2 * e1.pitch + n1 * sizeof(Scalar);

//   // (*ptrAddr(dphi, globalOffset));
//   Scalar rho_total = 0.0f;
//   for (int n = 0; n < dev_params.num_species; n++) {
//     rho_total += *ptrAddr(rho[n], globalOffset);
//   }
//   Scalar diff = *ptrAddr(mesh_ptrs.dV, globalOffset) * rho_total;
//   // dev_mesh.delta[0] * dev_mesh.delta[1] * rho_total;
//   if (n1 > dev_mesh.guard[0] + 1) {
//     diff -=
//         (*ptrAddr(e1, globalOffset + sizeof(Scalar)) *
//              *ptrAddr(mesh_ptrs.A1_e, globalOffset + sizeof(Scalar))
//              -
//          *ptrAddr(e1, globalOffset) *
//              *ptrAddr(mesh_ptrs.A1_e, globalOffset) +
//          *ptrAddr(e2, globalOffset + e2.pitch) *
//              *ptrAddr(mesh_ptrs.A2_e, globalOffset + e2.pitch) -
//          *ptrAddr(e2, globalOffset) *
//              *ptrAddr(mesh_ptrs.A2_e, globalOffset));
//   }
//   Scalar r0 = std::exp(dev_mesh.pos(0, n1, false));
//   Scalar r1s = std::exp(dev_mesh.pos(0, n1 + 1, false));
//   Scalar r1 = std::exp(dev_mesh.pos(0, n1, true));

//   Scalar Ar1 =
//       *ptrAddr(mesh_ptrs.A1_e, globalOffset + sizeof(Scalar)) / r1s;
//   Scalar Ar0 = *ptrAddr(mesh_ptrs.A1_e, globalOffset) / r0;
//   Scalar At1 = *ptrAddr(mesh_ptrs.A2_e, globalOffset + e2.pitch) /
//   r1; Scalar At0 = *ptrAddr(mesh_ptrs.A2_e, globalOffset) / r1;

//   Scalar Atot = (Ar1 + Ar0 + At1 + At0);

//   if (n1 > dev_mesh.guard[0] + 1) {
//     (*ptrAddr(dphi, globalOffset)) =
//         (diff + (Ar1 * *ptrAddr(dphi, globalOffset + sizeof(Scalar))
//         +
//                  Ar0 * *ptrAddr(dphi, globalOffset - sizeof(Scalar))
//                  + At1 * *ptrAddr(dphi, globalOffset + dphi.pitch) +
//                  At0 * *ptrAddr(dphi, globalOffset - dphi.pitch))) /
//         Atot;
//   }
// }

// __global__ void
// correct_E_field(pitchptr<Scalar e1, pitchptr<Scalar
// e2,
//                 pitchptr<Scalar dphi,
//                 Grid_LogSph_dev::mesh_ptrs mesh_ptrs) {
//   int t1 = blockIdx.x, t2 = blockIdx.y;
//   int c1 = threadIdx.x, c2 = threadIdx.y;
//   int n1 = dev_mesh.guard[0] + t1 * blockDim.x + c1;
//   int n2 = dev_mesh.guard[1] + t2 * blockDim.y + c2;
//   size_t globalOffset = n2 * e1.pitch + n1 * sizeof(Scalar);

//   if (n1 > dev_mesh.guard[0] + 1) {
//     (*ptrAddr(e1, globalOffset)) -=
//         (*ptrAddr(dphi, globalOffset) -
//          *ptrAddr(dphi, globalOffset - sizeof(Scalar))) /
//         (std::exp(dev_mesh.pos(0, n1, false)) * dev_mesh.delta[0]);
//     (*ptrAddr(e2, globalOffset)) -=
//         (*ptrAddr(dphi, globalOffset) -
//          *ptrAddr(dphi, globalOffset - dphi.pitch)) /
//         (std::exp(dev_mesh.pos(0, n1, true)) * dev_mesh.delta[1]);

//     if (blockIdx.y == 0 && threadIdx.y == 0) {
//       n2 = dev_mesh.guard[1] - 1;
//       globalOffset = n2 * e1.pitch + n1 * sizeof(Scalar);

//       (*ptrAddr(e1, globalOffset)) -=
//           (*ptrAddr(dphi, globalOffset) -
//            *ptrAddr(dphi, globalOffset - sizeof(Scalar))) /
//           (std::exp(dev_mesh.pos(0, n1, false)) * dev_mesh.delta[0]);
//     }
//   }
// }

}  // namespace Kernels

FieldSolver_LogSph::FieldSolver_LogSph() {}

FieldSolver_LogSph::~FieldSolver_LogSph() {}

void
FieldSolver_LogSph::update_fields(cu_sim_data &data, double dt,
                                  double time) {
  // Only implemented 2D!
  if (data.env.grid().dim() != 2) return;
  timer::stamp("field_update");

  // update_fields(data.E, data.B, data.J, dt, time);
  // Logger::print_info("Updating fields");
  data.env.get_sub_guard_cells(data.E);

  for_each_device(data.dev_map, [&data, dt, time](int n) {
    const Grid_LogSph_dev &grid =
        *dynamic_cast<const Grid_LogSph_dev *>(data.grid[n].get());
    auto mesh_ptrs = grid.get_mesh_ptrs();
    auto &mesh = grid.mesh();

    dim3 blockSize(32, 16);
    dim3 gridSize(mesh.reduced_dim(0) / 32, mesh.reduced_dim(1) / 16);
    // Update B
    Kernels::compute_b_update<<<gridSize, blockSize>>>(
        data.E[n].ptr(0), data.E[n].ptr(1), data.E[n].ptr(2),
        data.B[n].ptr(0), data.B[n].ptr(1), data.B[n].ptr(2), mesh_ptrs,
        dt);
    CudaCheckError();
  });

  data.env.get_sub_guard_cells(data.B);
  data.env.get_sub_guard_cells(data.J);
  for_each_device(data.dev_map, [&data, dt, time](int n) {
    const Grid_LogSph_dev &grid =
        *dynamic_cast<const Grid_LogSph_dev *>(data.grid[n].get());
    auto mesh_ptrs = grid.get_mesh_ptrs();
    auto &mesh = grid.mesh();

    dim3 blockSize(32, 16);
    dim3 gridSize(mesh.reduced_dim(0) / 32, mesh.reduced_dim(1) / 16);
    // Update B
    Kernels::compute_e_update<<<gridSize, blockSize>>>(
        data.E[n].ptr(0), data.E[n].ptr(1), data.E[n].ptr(2),
        data.B[n].ptr(0), data.B[n].ptr(1), data.B[n].ptr(2),
        data.J[n].ptr(0), data.J[n].ptr(1), data.J[n].ptr(2),
        data.Rho[0][n].ptr(), data.Rho[1][n].ptr(),
        data.Rho[2][n].ptr(), mesh_ptrs, dt);
    CudaCheckError();
  });

  data.env.get_sub_guard_cells(data.E);

  for_each_device(data.dev_map, [&data](int n) {
    const Grid_LogSph_dev &grid =
        *dynamic_cast<const Grid_LogSph_dev *>(data.grid[n].get());
    auto mesh_ptrs = grid.get_mesh_ptrs();
    auto &mesh = grid.mesh();

    dim3 blockSize(32, 16);
    dim3 gridSize(mesh.reduced_dim(0) / 32, mesh.reduced_dim(1) / 16);
    // Update B
    Kernels::compute_divs<<<gridSize, blockSize>>>(
        data.E[n].ptr(0), data.E[n].ptr(1), data.E[n].ptr(2),
        data.B[n].ptr(0), data.B[n].ptr(1), data.B[n].ptr(2),
        data.divE[n].ptr(), data.divB[n].ptr(), mesh_ptrs);
    CudaCheckError();
  });
  data.compute_edotb();

  CudaSafeCall(cudaDeviceSynchronize());
  timer::show_duration_since_stamp("Field update", "us",
                                   "field_update");
}

void
FieldSolver_LogSph::set_background_j(const vfield_t &J) {}

void
FieldSolver_LogSph::boundary_conditions(cu_sim_data &data,
                                        double omega) {
  for (int n = 0; n < data.dev_map.size(); n++) {
    int dev_id = data.dev_map[n];
    CudaSafeCall(cudaSetDevice(dev_id));
    if (data.env.is_boundary(n, (int)BoundaryPos::lower0)) {
      Kernels::stellar_boundary<<<32, 256>>>(
          data.E[n].ptr(0), data.E[n].ptr(1), data.E[n].ptr(2),
          data.B[n].ptr(0), data.B[n].ptr(1), data.B[n].ptr(2), omega);
      CudaCheckError();
    }

    if (data.env.is_boundary(n, (int)BoundaryPos::upper0)) {
      Kernels::outflow_boundary<<<32, 256>>>(
          data.E[n].ptr(0), data.E[n].ptr(1), data.E[n].ptr(2),
          data.B[n].ptr(0), data.B[n].ptr(1), data.B[n].ptr(2));
      CudaCheckError();
    }

    if (data.env.is_boundary(n, (int)BoundaryPos::lower1)) {
      Kernels::axis_boundary_lower<<<32, 256>>>(
          data.E[n].ptr(0), data.E[n].ptr(1), data.E[n].ptr(2),
          data.B[n].ptr(0), data.B[n].ptr(1), data.B[n].ptr(2));
      CudaCheckError();
    }

    if (data.env.is_boundary(n, (int)BoundaryPos::upper1)) {
      Kernels::axis_boundary_upper<<<32, 256>>>(
          data.E[n].ptr(0), data.E[n].ptr(1), data.E[n].ptr(2),
          data.B[n].ptr(0), data.B[n].ptr(1), data.B[n].ptr(2));
      CudaCheckError();
    }
  }
  // Logger::print_info("omega is {}", omega);
}

}  // namespace Aperture
