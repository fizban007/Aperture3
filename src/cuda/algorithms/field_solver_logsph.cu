#include "algorithms/field_solver_logsph.h"
#include "cuda/constant_mem.h"
#include "cuda/cudaUtility.h"
#include "cuda/data_ptrs.h"
#include "cuda/grids/grid_log_sph_ptrs.h"
#include "cuda/utils/pitchptr.h"
#include "grids/grid_log_sph.h"
#include "sim_data.h"
#include "sim_environment.h"
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
compute_e_update_logsph(data_ptrs data,
                        mesh_ptrs_log_sph mesh_ptrs, Scalar dt) {
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
    data.E1[globalOffset] += dt * (-4.0f * data.B3[globalOffset] * alpha_gr(r0) /
                                  (dev_mesh.delta[1] * r0) -
                              // alpha_gr(r0) * j1[globalOffset]);
                              data.J1[globalOffset]);
  } else {
    data.E1[globalOffset] +=
        // -dt * j1[globalOffset];
        dt *
        ((data.B3(n1, n2 + 1) * alpha_gr(r0) * mesh_ptrs.l3_b(n1, n2 + 1) -
          data.B3(n1, n2) * alpha_gr(r0) * mesh_ptrs.l3_b(n1, n2)) /
             mesh_ptrs.A1_e(n1, n2) -
         // alpha_gr(r0) * j1(n1, n2));
         data.J1(n1, n2));
  }
  // (Curl u)_2 = d3u1 - d1u3
  data.E2[globalOffset] +=
      // -dt * j2[globalOffset];
      dt *
      ((data.B3(n1, n2) * alpha_gr(r0) * mesh_ptrs.l3_b(n1, n2) -
        data.B3(n1 + 1, n2) * alpha_gr(r1) * mesh_ptrs.l3_b(n1 + 1, n2)) /
           mesh_ptrs.A2_e(n1, n2) -
       // alpha_gr(r) * j2(n1, n2));
       data.J2(n1, n2));

  // (Curl u)_3 = d1u2 - d2u1
  data.E3[globalOffset] +=
      // -dt * j3[globalOffset];
      dt * ((data.B2(n1 + 1, n2) * alpha_gr(r1) *
                 // e1(n1 + 1, n2) * beta_phi(r1, theta)) *
                 mesh_ptrs.l2_b(n1 + 1, n2) -
             data.B2(n1, n2) * alpha_gr(r0) *
                 // - e1(n1, n2) * beta_phi(r0, theta)) *
                 mesh_ptrs.l2_b(n1, n2) +
             data.B1(n1, n2) * alpha_gr(r) *
                 // + e2(n1, n2) * beta_phi(r, theta0)) *
                 mesh_ptrs.l1_b(n1, n2) -
             data.B1(n1, n2 + 1) * alpha_gr(r) *
                 // e2(n1, n2 + 1) * beta_phi(r, theta0 +
                 // dev_mesh.delta[1])) *
                 mesh_ptrs.l1_b(n1, n2 + 1)) /
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

    data.E3[globalOffset] = 0.0f;

    data.E1[globalOffset] += dt * (4.0f * data.B3(n1, n2 + 1) * alpha_gr(r0) /
                                  (dev_mesh.delta[1] * r0) -
                              // alpha_gr(r0) * j1[globalOffset]);
                              data.J1[globalOffset]);
  }
}

// template <int DIM1, int DIM2>
__global__ void
compute_b_update_logsph(data_ptrs data,
                        mesh_ptrs_log_sph mesh_ptrs, Scalar dt) {
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
      (data.E3(n1, n2) * alpha_gr(r1) * mesh_ptrs.l3_e(n1, n2) -
       data.E3(n1, n2 - 1) * alpha_gr(r1) * mesh_ptrs.l3_e(n1, n2 - 1)) /
      mesh_ptrs.A1_b(n1, n2);

  // (Curl u)_2 = d3u1 - d1u3
  data.B2[globalOffset] +=
      -dt *
      (data.E3(n1 - 1, n2) * alpha_gr(r0) * mesh_ptrs.l3_e(n1 - 1, n2) -
       data.E3(n1, n2) * alpha_gr(r1) * mesh_ptrs.l3_e(n1, n2)) /
      mesh_ptrs.A2_b(n1, n2);

  // (Curl u)_3 = d1u2 - d2u1
  data.B3[globalOffset] +=
      -dt * (((data.E2(n1, n2) * alpha_gr(r1) +
               // (b1(n1, n2) + dev_bg_fields.B1(n1, n2)) * beta_phi(r1,
               // dev_mesh.pos(1, n2, 0))) *
               data.Bbg1(n1, n2) *
                   beta_phi(r1, dev_mesh.pos(1, n2, 0))) *
                  mesh_ptrs.l2_e(n1, n2) -
              (data.E2(n1 - 1, n2) * alpha_gr(r0) +
               // (b1(n1 - 1, n2) + dev_bg_fields.B1(n1 - 1, n2)) *
               // beta_phi(r0, dev_mesh.pos(1, n2, 0))) *
               data.Bbg1(n1 - 1, n2) *
                   beta_phi(r0, dev_mesh.pos(1, n2, 0))) *
                  mesh_ptrs.l2_e(n1 - 1, n2) +
              (data.E1(n1, n2 - 1) * alpha_gr(r) -
               // (b2(n1, n2 - 1) + dev_bg_fields.B2(n1, n2 - 1)) *
               // beta_phi(r, dev_mesh.pos(1, n2 - 1, 1))) *
               data.Bbg2(n1, n2 - 1) *
                   beta_phi(r, dev_mesh.pos(1, n2 - 1, 1))) *
                  mesh_ptrs.l1_e(n1, n2 - 1) -
              (data.E1(n1, n2) * alpha_gr(r) -
               // (b2(n1, n2) + dev_bg_fields.B2(n1, n2)) * beta_phi(r,
               // dev_mesh.pos(1, n2, 1))) *
               data.Bbg2(n1, n2) *
                   beta_phi(r, dev_mesh.pos(1, n2, 1))) *
                  mesh_ptrs.l1_e(n1, n2)) /
             mesh_ptrs.A3_b(n1, n2));

  __syncthreads();

  // Extra work for the axis at theta = 0
  if (std::abs(dev_mesh.pos(1, n2, true) - dev_mesh.delta[1]) <
      1.0e-5) {
    n2 = dev_mesh.guard[1] - 1;
    globalOffset = data.B2.compute_offset(n1, n2);

    data.B2[globalOffset] = 0.0f;
  }
}

// template <int DIM1, int DIM2>
__global__ void
compute_divs_logsph(pitchptr<Scalar> e1, pitchptr<Scalar> e2,
                    pitchptr<Scalar> e3, pitchptr<Scalar> b1,
                    pitchptr<Scalar> b2, pitchptr<Scalar> b3,
                    pitchptr<Scalar> divE, pitchptr<Scalar> divB,
                    mesh_ptrs_log_sph mesh_ptrs) {
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
                 pitchptr<Scalar> b01, pitchptr<Scalar> b02,
                 pitchptr<Scalar> b03, Scalar omega) {
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
      e2(i, j) = -(omega - omega_LT) * std::sin(theta) * b01(i, j) /
                 alpha_gr(r_s) / r_s / r_s;
      e1(i, j) = (omega - omega_LT) * std::sin(theta_s) * b02(i, j) /
                 alpha_gr(r_s) / r / r;
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
outflow_boundary_sph(pitchptr<Scalar> e1, pitchptr<Scalar> e2,
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

field_solver_logsph::field_solver_logsph(sim_environment &env)
    : m_env(env) {}

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
  auto data_ptrs = get_data_ptrs(data);

  dim3 blockSize(32, 16);
  dim3 gridSize(mesh.reduced_dim(0) / 32, mesh.reduced_dim(1) / 16);
  // Update B
  Kernels::compute_b_update_logsph<<<gridSize, blockSize>>>(
      data_ptrs, mesh_ptrs, dt);
  CudaCheckError();

  // Communicate the new B values to guard cells
  m_env.send_guard_cells(data.B);
  // m_env.send_guard_cells(data.J);

  // Update E
  Kernels::compute_e_update_logsph<<<gridSize, blockSize>>>(
      data_ptrs, mesh_ptrs, dt);
  CudaCheckError();

  // Communicate the new E values to guard cells
  m_env.send_guard_cells(data.E);

  // Compute divergences
  Kernels::compute_divs_logsph<<<gridSize, blockSize>>>(
      get_pitchptr(data.E.data(0)), get_pitchptr(data.E.data(1)),
      get_pitchptr(data.E.data(2)), get_pitchptr(data.B.data(0)),
      get_pitchptr(data.B.data(1)), get_pitchptr(data.B.data(2)),
      get_pitchptr(data.divE.data()), get_pitchptr(data.divB.data()),
      mesh_ptrs);
  CudaCheckError();
  data.compute_edotb();

  CudaSafeCall(cudaDeviceSynchronize());
  timer::show_duration_since_stamp("Field update", "us",
                                   "field_update");
}

// void
// field_solver_logsph::set_background_j(const vfield_t &J) {}

void
field_solver_logsph::apply_boundary(sim_data &data, double omega,
                                    double time) {
  // int dev_id = data.dev_id;
  // CudaSafeCall(cudaSetDevice(dev_id));
  if (data.env.is_boundary(BoundaryPos::lower0)) {
    Kernels::stellar_boundary<<<32, 256>>>(
        get_pitchptr(data.E.data(0)), get_pitchptr(data.E.data(1)),
        get_pitchptr(data.E.data(2)), get_pitchptr(data.B.data(0)),
        get_pitchptr(data.B.data(1)), get_pitchptr(data.B.data(2)),
        get_pitchptr(data.Bbg.data(0)), get_pitchptr(data.Bbg.data(1)),
        get_pitchptr(data.Bbg.data(2)), omega);
    CudaCheckError();
  }

  if (data.env.is_boundary(BoundaryPos::upper0)) {
    Kernels::outflow_boundary_sph<<<32, 256>>>(
        get_pitchptr(data.E.data(0)), get_pitchptr(data.E.data(1)),
        get_pitchptr(data.E.data(2)), get_pitchptr(data.B.data(0)),
        get_pitchptr(data.B.data(1)), get_pitchptr(data.B.data(2)));
    CudaCheckError();
  }

  if (data.env.is_boundary(BoundaryPos::lower1)) {
    Kernels::axis_boundary_lower<<<32, 256>>>(
        get_pitchptr(data.E.data(0)), get_pitchptr(data.E.data(1)),
        get_pitchptr(data.E.data(2)), get_pitchptr(data.B.data(0)),
        get_pitchptr(data.B.data(1)), get_pitchptr(data.B.data(2)));
    CudaCheckError();
  }

  if (data.env.is_boundary(BoundaryPos::upper1)) {
    Kernels::axis_boundary_upper<<<32, 256>>>(
        get_pitchptr(data.E.data(0)), get_pitchptr(data.E.data(1)),
        get_pitchptr(data.E.data(2)), get_pitchptr(data.B.data(0)),
        get_pitchptr(data.B.data(1)), get_pitchptr(data.B.data(2)));
    CudaCheckError();
  }
  // Logger::print_info("omega is {}", omega);
}

}  // namespace Aperture
