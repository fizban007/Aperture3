#include "algorithms/field_solver.h"
#include "cuda/constant_mem.h"
#include "cuda/cudaUtility.h"
#include "cuda/data_ptrs.h"
#include "cuda/utils/pitchptr.h"
#include "sim_data.h"
#include "sim_environment.h"
#include "utils/timer.h"

namespace Aperture {

namespace Kernels {

__global__ void
compute_e_update1d(pitchptr<Scalar> e1, pitchptr<Scalar> e2,
                   pitchptr<Scalar> e3, pitchptr<Scalar> b1,
                   pitchptr<Scalar> b2, pitchptr<Scalar> b3,
                   pitchptr<Scalar> b01, pitchptr<Scalar> b02,
                   pitchptr<Scalar> b03, pitchptr<Scalar> j1,
                   pitchptr<Scalar> j2, pitchptr<Scalar> j3,
                   Scalar dt) {
  // Load position parameters
  int t1 = blockIdx.x;
  int c1 = threadIdx.x;
  int n1 = dev_mesh.guard[0] + t1 * blockDim.x + c1;
  if (n1 >= dev_mesh.dims[0]) return;
  size_t globalOffset = e1.compute_offset(n1);

  // Do the actual computation here
  // (Curl u)_1 = d2u3 - d3u2
  e1[globalOffset] -= dt * j1(n1);
  // (Curl u)_2 = d3u1 - d1u3
  e2[globalOffset] +=
      dt * ((b3(n1) - b3(n1 + 1) - b03(n1) + b03(n1 + 1)) *
                dev_mesh.inv_delta[0] -
            j2(n1));

  // (Curl u)_3 = d1u2 - d2u1
  e3[globalOffset] +=
      dt * ((b2(n1 + 1) - b2(n1) - b02(n1 + 1) + b02(n1)) *
                dev_mesh.inv_delta[0] -
            j3(n1));
}

__global__ void
compute_e_update2d(pitchptr<Scalar> e1, pitchptr<Scalar> e2,
                   pitchptr<Scalar> e3, pitchptr<Scalar> b1,
                   pitchptr<Scalar> b2, pitchptr<Scalar> b3,
                   pitchptr<Scalar> b01, pitchptr<Scalar> b02,
                   pitchptr<Scalar> b03, pitchptr<Scalar> j1,
                   pitchptr<Scalar> j2, pitchptr<Scalar> j3,
                   Scalar dt) {
  // Load position parameters
  int t1 = blockIdx.x, t2 = blockIdx.y;
  int c1 = threadIdx.x, c2 = threadIdx.y;
  int n1 = dev_mesh.guard[0] + t1 * blockDim.x + c1;
  int n2 = dev_mesh.guard[1] + t2 * blockDim.y + c2;
  if (n1 >= dev_mesh.dims[0] || n2 >= dev_mesh.dims[1]) return;
  size_t globalOffset = e1.compute_offset(n1, n2);

  // Do the actual computation here
  // (Curl u)_1 = d2u3 - d3u2
  e1[globalOffset] +=
      dt *
      ((b3(n1, n2 + 1) - b3(n1, n2) - b03(n1, n2 + 1) + b03(n1, n2)) *
           dev_mesh.inv_delta[1] -
       j1(n1, n2));

  // (Curl u)_2 = d3u1 - d1u3
  e2[globalOffset] +=
      dt *
      ((b3(n1, n2) - b3(n1 + 1, n2) - b03(n1, n2) + b03(n1 + 1, n2)) *
           dev_mesh.inv_delta[0] -
       j2(n1, n2));

  // (Curl u)_3 = d1u2 - d2u1
  e3[globalOffset] +=
      dt *
      ((b2(n1 + 1, n2) - b2(n1, n2) - b02(n1 + 1, n2) + b02(n1, n2)) *
           dev_mesh.inv_delta[0] +
       (b1(n1, n2) - b1(n1, n2 + 1) - b01(n1, n2) + b01(n1, n2 + 1)) *
           dev_mesh.inv_delta[1] -
       j3(n1, n2));
}

__global__ void
compute_e_update3d(pitchptr<Scalar> e1, pitchptr<Scalar> e2,
                   pitchptr<Scalar> e3, pitchptr<Scalar> b1,
                   pitchptr<Scalar> b2, pitchptr<Scalar> b3,
                   pitchptr<Scalar> b01, pitchptr<Scalar> b02,
                   pitchptr<Scalar> b03, pitchptr<Scalar> j1,
                   pitchptr<Scalar> j2, pitchptr<Scalar> j3,
                   Scalar dt) {
  // Load position parameters
  int t1 = blockIdx.x, t2 = blockIdx.y, t3 = blockIdx.z;
  int c1 = threadIdx.x, c2 = threadIdx.y, c3 = threadIdx.z;
  int n1 = dev_mesh.guard[0] + t1 * blockDim.x + c1;
  int n2 = dev_mesh.guard[1] + t2 * blockDim.y + c2;
  int n3 = dev_mesh.guard[2] + t3 * blockDim.z + c3;
  if (n1 >= dev_mesh.dims[0] || n2 >= dev_mesh.dims[1] ||
      n3 >= dev_mesh.dims[2])
    return;
  size_t globalOffset = e1.compute_offset(n1, n2, n3);

  // Do the actual computation here
  // (Curl u)_1 = d2u3 - d3u2
  e1[globalOffset] += dt * ((b3(n1, n2 + 1, n3) - b3(n1, n2, n3) -
                             b03(n1, n2 + 1, n3) + b03(n1, n2, n3)) *
                                dev_mesh.inv_delta[1] +
                            (b2(n1, n2, n3) - b2(n1, n2, n3 + 1) -
                             b02(n1, n2, n3) + b02(n1, n2, n3 + 1)) *
                                dev_mesh.inv_delta[2] -
                            j1(n1, n2, n3));
  // (Curl u)_2 = d3u1 - d1u3
  e2[globalOffset] += dt * ((b3(n1, n2, n3) - b3(n1 + 1, n2, n3) -
                             b03(n1, n2, n3) + b03(n1 + 1, n2, n3)) *
                                dev_mesh.inv_delta[0] +
                            (b1(n1, n2, n3 + 1) - b1(n1, n2, n3) -
                             b01(n1, n2, n3 + 1) + b01(n1, n2, n3)) *
                                dev_mesh.inv_delta[2] -
                            j2(n1, n2, n3));

  // (Curl u)_3 = d1u2 - d2u1
  e3[globalOffset] += dt * ((b2(n1 + 1, n2, n3) - b2(n1, n2, n3) -
                             b02(n1 + 1, n2, n3) + b02(n1, n2, n3)) *
                                dev_mesh.inv_delta[0] +
                            (b1(n1, n2, n3) - b1(n1, n2 + 1, n3) -
                             b01(n1, n2, n3) + b01(n1, n2 + 1, n3)) *
                                dev_mesh.inv_delta[1] -
                            j3(n1, n2, n3));
}

__global__ void
compute_b_update1d(pitchptr<Scalar> e1, pitchptr<Scalar> e2,
                   pitchptr<Scalar> e3, pitchptr<Scalar> e01,
                   pitchptr<Scalar> e02, pitchptr<Scalar> e03,
                   pitchptr<Scalar> b1, pitchptr<Scalar> b2,
                   pitchptr<Scalar> b3, Scalar dt) {
  // Load position parameters
  int t1 = blockIdx.x;
  int c1 = threadIdx.x;
  int n1 = dev_mesh.guard[0] + t1 * blockDim.x + c1;
  if (n1 >= dev_mesh.dims[0]) return;
  size_t globalOffset = e1.compute_offset(n1);

  // Do the actual computation here
  // (Curl u)_1 = d2u3 - d3u2
  // b1 does not change in 1d
  // (Curl u)_2 = d3u1 - d1u3
  b2[globalOffset] -= dt *
                      (e3(n1 - 1) - e3(n1) - e03(n1 - 1) + e03(n1)) *
                      dev_mesh.inv_delta[0];

  // (Curl u)_3 = d1u2 - d2u1
  b3[globalOffset] -= dt *
                      (e2(n1) - e2(n1 - 1) - e02(n1) + e02(n1 - 1)) *
                      dev_mesh.inv_delta[0];
}

__global__ void
compute_b_update2d(pitchptr<Scalar> e1, pitchptr<Scalar> e2,
                   pitchptr<Scalar> e3, pitchptr<Scalar> e01,
                   pitchptr<Scalar> e02, pitchptr<Scalar> e03,
                   pitchptr<Scalar> b1, pitchptr<Scalar> b2,
                   pitchptr<Scalar> b3, Scalar dt) {
  // Load position parameters
  int t1 = blockIdx.x, t2 = blockIdx.y;
  int c1 = threadIdx.x, c2 = threadIdx.y;
  int n1 = dev_mesh.guard[0] + t1 * blockDim.x + c1;
  int n2 = dev_mesh.guard[1] + t2 * blockDim.y + c2;
  if (n1 >= dev_mesh.dims[0] || n2 >= dev_mesh.dims[1]) return;
  size_t globalOffset = e1.compute_offset(n1, n2);

  // Do the actual computation here
  // (Curl u)_1 = d2u3 - d3u2
  b1[globalOffset] -=
      dt *
      (e3(n1, n2) - e3(n1, n2 - 1) - e03(n1, n2) + e03(n1, n2 - 1)) *
      dev_mesh.inv_delta[1];
  // (Curl u)_2 = d3u1 - d1u3
  b2[globalOffset] -=
      dt *
      (e3(n1 - 1, n2) - e3(n1, n2) - e03(n1 - 1, n2) + e03(n1, n2)) *
      dev_mesh.inv_delta[0];

  // (Curl u)_3 = d1u2 - d2u1
  b3[globalOffset] -=
      dt *
      ((e2(n1, n2) - e2(n1 - 1, n2) - e02(n1, n2) + e02(n1 - 1, n2)) *
           dev_mesh.inv_delta[0] +
       (e1(n1, n2 - 1) - e1(n1, n2) - e01(n1, n2 - 1) + e01(n1, n2)) *
           dev_mesh.inv_delta[1]);
}

__global__ void
compute_b_update3d(pitchptr<Scalar> e1, pitchptr<Scalar> e2,
                   pitchptr<Scalar> e3, pitchptr<Scalar> e01,
                   pitchptr<Scalar> e02, pitchptr<Scalar> e03,
                   pitchptr<Scalar> b1, pitchptr<Scalar> b2,
                   pitchptr<Scalar> b3, Scalar dt) {
  // Load position parameters
  int t1 = blockIdx.x, t2 = blockIdx.y, t3 = blockIdx.z;
  int c1 = threadIdx.x, c2 = threadIdx.y, c3 = threadIdx.z;
  int n1 = dev_mesh.guard[0] + t1 * blockDim.x + c1;
  int n2 = dev_mesh.guard[1] + t2 * blockDim.y + c2;
  int n3 = dev_mesh.guard[2] + t3 * blockDim.z + c3;
  if (n1 >= dev_mesh.dims[0] || n2 >= dev_mesh.dims[1] ||
      n3 >= dev_mesh.dims[2])
    return;
  size_t globalOffset = e1.compute_offset(n1, n2, n3);

  // Do the actual computation here
  // (Curl u)_1 = d2u3 - d3u2
  b1[globalOffset] -= dt * ((e3(n1, n2 + 1, n3) - e3(n1, n2, n3) -
                             e03(n1, n2 + 1, n3) + e03(n1, n2, n3)) *
                                dev_mesh.inv_delta[1] +
                            (e2(n1, n2, n3) - e2(n1, n2, n3 + 1) -
                             e02(n1, n2, n3) + e02(n1, n2, n3 + 1)) *
                                dev_mesh.inv_delta[2]);
  // (Curl u)_2 = d3u1 - d1u3
  b2[globalOffset] -= dt * ((e3(n1, n2, n3) - e3(n1 + 1, n2, n3) -
                             e03(n1, n2, n3) + e03(n1 + 1, n2, n3)) *
                                dev_mesh.inv_delta[0] +
                            (e1(n1, n2, n3 + 1) - e1(n1, n2, n3) -
                             e01(n1, n2, n3 + 1) + e01(n1, n2, n3)) *
                                dev_mesh.inv_delta[2]);

  // (Curl u)_3 = d1u2 - d2u1
  b3[globalOffset] -= dt * ((e2(n1 + 1, n2, n3) - e2(n1, n2, n3) -
                             e02(n1 + 1, n2, n3) + e02(n1, n2, n3)) *
                                dev_mesh.inv_delta[0] +
                            (e1(n1, n2, n3) - e1(n1, n2 + 1, n3) -
                             e01(n1, n2, n3) + e01(n1, n2 + 1, n3)) *
                                dev_mesh.inv_delta[1]);
}

__global__ void
compute_divs_1d(pitchptr<Scalar> e1, pitchptr<Scalar> e2,
                pitchptr<Scalar> e3, pitchptr<Scalar> b1,
                pitchptr<Scalar> b2, pitchptr<Scalar> b3,
                pitchptr<Scalar> e01, pitchptr<Scalar> e02,
                pitchptr<Scalar> e03, pitchptr<Scalar> b01,
                pitchptr<Scalar> b02, pitchptr<Scalar> b03,
                pitchptr<Scalar> divE, pitchptr<Scalar> divB) {
  int t1 = blockIdx.x;
  int c1 = threadIdx.x;
  int n1 = dev_mesh.guard[0] + t1 * blockDim.x + c1;
  if (n1 >= dev_mesh.dims[0]) return;

  size_t globalOffset = divE.compute_offset(n1);

  divE[globalOffset] =
      (e1(n1 + 1) - e01(n1 + 1) - e1(n1) + e01(n1)) *
          dev_mesh.inv_delta[0];

  divB[globalOffset] =
      (b1(n1) - b01(n1) - b1(n1 - 1) + b01(n1 - 1)) *
          dev_mesh.inv_delta[0];
}

__global__ void
compute_divs_2d(pitchptr<Scalar> e1, pitchptr<Scalar> e2,
                pitchptr<Scalar> e3, pitchptr<Scalar> b1,
                pitchptr<Scalar> b2, pitchptr<Scalar> b3,
                pitchptr<Scalar> e01, pitchptr<Scalar> e02,
                pitchptr<Scalar> e03, pitchptr<Scalar> b01,
                pitchptr<Scalar> b02, pitchptr<Scalar> b03,
                pitchptr<Scalar> divE, pitchptr<Scalar> divB) {
  int t1 = blockIdx.x, t2 = blockIdx.y;
  int c1 = threadIdx.x, c2 = threadIdx.y;
  int n1 = dev_mesh.guard[0] + t1 * blockDim.x + c1;
  int n2 = dev_mesh.guard[1] + t2 * blockDim.y + c2;
  if (n1 >= dev_mesh.dims[0] || n2 >= dev_mesh.dims[1]) return;

  size_t globalOffset = divE.compute_offset(n1, n2);

  divE[globalOffset] =
      (e1(n1 + 1, n2) - e01(n1 + 1, n2) - e1(n1, n2) + e01(n1, n2)) *
          dev_mesh.inv_delta[0] +
      (e2(n1, n2 + 1) - e02(n1, n2 + 1) - e2(n1, n2) + e02(n1, n2)) *
          dev_mesh.inv_delta[1];

  divB[globalOffset] =
      (b1(n1, n2) - b01(n1, n2) - b1(n1 - 1, n2) + b01(n1 - 1, n2)) *
          dev_mesh.inv_delta[0] +
      (b2(n1, n2) - b02(n1, n2) - b2(n1, n2 - 1) + b02(n1, n2 - 1)) *
          dev_mesh.inv_delta[1];
}

__global__ void
compute_divs_3d(pitchptr<Scalar> e1, pitchptr<Scalar> e2,
                pitchptr<Scalar> e3, pitchptr<Scalar> b1,
                pitchptr<Scalar> b2, pitchptr<Scalar> b3,
                pitchptr<Scalar> e01, pitchptr<Scalar> e02,
                pitchptr<Scalar> e03, pitchptr<Scalar> b01,
                pitchptr<Scalar> b02, pitchptr<Scalar> b03,
                pitchptr<Scalar> divE, pitchptr<Scalar> divB) {
  // Load position parameters
  int t1 = blockIdx.x, t2 = blockIdx.y, t3 = blockIdx.z;
  int c1 = threadIdx.x, c2 = threadIdx.y, c3 = threadIdx.z;
  int n1 = dev_mesh.guard[0] + t1 * blockDim.x + c1;
  int n2 = dev_mesh.guard[1] + t2 * blockDim.y + c2;
  int n3 = dev_mesh.guard[2] + t3 * blockDim.z + c3;
  if (n1 >= dev_mesh.dims[0] || n2 >= dev_mesh.dims[1] ||
      n3 >= dev_mesh.dims[2])
    return;

  size_t globalOffset = divE.compute_offset(n1, n2, n3);

  // if (n1 > dev_mesh.guard[0] + 1) {
  divE[globalOffset] =
      (e1(n1 + 1, n2, n3) - e01(n1 + 1, n2, n3) - e1(n1, n2, n3) + e01(n1, n2, n3)) *
          dev_mesh.inv_delta[0] +
      (e2(n1, n2 + 1, n3) - e02(n1, n2 + 1, n3) - e2(n1, n2, n3) + e02(n1, n2, n3)) *
          dev_mesh.inv_delta[1] +
      (e3(n1, n2, n3 + 1) - e03(n1, n2, n3 + 1) - e3(n1, n2, n3) + e03(n1, n2, n3)) *
          dev_mesh.inv_delta[2];

  divB[globalOffset] =
      (b1(n1, n2, n3) - b01(n1, n2, n3) - b1(n1 - 1, n2, n3) + b01(n1 - 1, n2, n3)) *
          dev_mesh.inv_delta[0] +
      (b2(n1, n2, n3) - b02(n1, n2, n3) - b2(n1, n2 - 1, n3) + b02(n1, n2 - 1, n3)) *
          dev_mesh.inv_delta[1] +
      (b3(n1, n2, n3) - b03(n1, n2, n3) - b3(n1, n2, n3 - 1) + b03(n1, n2, n3 - 1)) *
          dev_mesh.inv_delta[2];
}

}  // namespace Kernels

field_solver::field_solver(sim_environment &env) : m_env(env) {}

field_solver::~field_solver() {}

void
field_solver::update_fields(sim_data &data, double dt, uint32_t step) {
  timer::stamp("field_update");
  if (step == 0) {
    update_e_field(data, 0.5 * dt);
    m_env.send_guard_cells(data.E);
  }

  update_b_field(data, dt);
  // Communicate the new B values to guard cells
  timer::stamp("send_B");
  m_env.send_guard_cells(data.B);
  timer::show_duration_since_stamp("Sending B field", "ms",
                                   "send_B");

  update_e_field(data, dt);
  // Communicate the new E values to guard cells
  m_env.send_guard_cells(data.E);

  compute_divs(data);

  CudaSafeCall(cudaDeviceSynchronize());
  timer::show_duration_since_stamp("Field update", "ms",
                                   "field_update");
}

void
field_solver::update_b_field(sim_data &data, double dt) {
  dim3 blockSize, gridSize;
  auto &grid = m_env.grid();
  auto &mesh = grid.mesh();
  if (grid.dim() == 1) {
    blockSize = dim3(512);
    gridSize =
        dim3((mesh.reduced_dim(0) + blockSize.x - 1) / blockSize.x);
    Kernels::compute_b_update1d<<<gridSize, blockSize>>>(
        get_pitchptr(data.E.data(0)), get_pitchptr(data.E.data(1)),
        get_pitchptr(data.E.data(2)), get_pitchptr(data.Ebg.data(0)),
        get_pitchptr(data.Ebg.data(1)), get_pitchptr(data.Ebg.data(2)),
        get_pitchptr(data.B.data(0)), get_pitchptr(data.B.data(1)),
        get_pitchptr(data.B.data(2)), dt);
  } else if (grid.dim() == 2) {
    blockSize = dim3(32, 16);
    gridSize =
        dim3((mesh.reduced_dim(0) + blockSize.x - 1) / blockSize.x,
             (mesh.reduced_dim(1) + blockSize.y - 1) / blockSize.y);

    Kernels::compute_b_update2d<<<gridSize, blockSize>>>(
        get_pitchptr(data.E.data(0)), get_pitchptr(data.E.data(1)),
        get_pitchptr(data.E.data(2)), get_pitchptr(data.Ebg.data(0)),
        get_pitchptr(data.Ebg.data(1)), get_pitchptr(data.Ebg.data(2)),
        get_pitchptr(data.B.data(0)), get_pitchptr(data.B.data(1)),
        get_pitchptr(data.B.data(2)), dt);
  } else if (grid.dim() == 3) {
    blockSize = dim3(32, 8, 4);
    gridSize =
        dim3((mesh.reduced_dim(0) + blockSize.x - 1) / blockSize.x,
             (mesh.reduced_dim(1) + blockSize.y - 1) / blockSize.y,
             (mesh.reduced_dim(2) + blockSize.z - 1) / blockSize.z);

    Kernels::compute_b_update3d<<<gridSize, blockSize>>>(
        get_pitchptr(data.E.data(0)), get_pitchptr(data.E.data(1)),
        get_pitchptr(data.E.data(2)), get_pitchptr(data.Ebg.data(0)),
        get_pitchptr(data.Ebg.data(1)), get_pitchptr(data.Ebg.data(2)),
        get_pitchptr(data.B.data(0)), get_pitchptr(data.B.data(1)),
        get_pitchptr(data.B.data(2)), dt);
  }
  CudaCheckError();
}

void
field_solver::update_e_field(sim_data &data, double dt) {
  dim3 blockSize, gridSize;
  auto &grid = m_env.grid();
  auto &mesh = grid.mesh();
  if (grid.dim() == 1) {
    blockSize = dim3(512);
    gridSize =
        dim3((mesh.reduced_dim(0) + blockSize.x - 1) / blockSize.x);
    Kernels::compute_e_update1d<<<gridSize, blockSize>>>(
        get_pitchptr(data.E.data(0)), get_pitchptr(data.E.data(1)),
        get_pitchptr(data.E.data(2)), get_pitchptr(data.B.data(0)),
        get_pitchptr(data.B.data(1)), get_pitchptr(data.B.data(2)),
        get_pitchptr(data.Bbg.data(0)), get_pitchptr(data.Bbg.data(1)),
        get_pitchptr(data.Bbg.data(2)), get_pitchptr(data.J.data(0)),
        get_pitchptr(data.J.data(1)), get_pitchptr(data.J.data(2)), dt);
  } else if (grid.dim() == 2) {
    blockSize = dim3(32, 16);
    gridSize =
        dim3((mesh.reduced_dim(0) + blockSize.x - 1) / blockSize.x,
             (mesh.reduced_dim(1) + blockSize.y - 1) / blockSize.y);

    Kernels::compute_e_update2d<<<gridSize, blockSize>>>(
        get_pitchptr(data.E.data(0)), get_pitchptr(data.E.data(1)),
        get_pitchptr(data.E.data(2)), get_pitchptr(data.B.data(0)),
        get_pitchptr(data.B.data(1)), get_pitchptr(data.B.data(2)),
        get_pitchptr(data.Bbg.data(0)), get_pitchptr(data.Bbg.data(1)),
        get_pitchptr(data.Bbg.data(2)), get_pitchptr(data.J.data(0)),
        get_pitchptr(data.J.data(1)), get_pitchptr(data.J.data(2)), dt);
  } else if (grid.dim() == 3) {
    blockSize = dim3(32, 8, 4);
    gridSize =
        dim3((mesh.reduced_dim(0) + blockSize.x - 1) / blockSize.x,
             (mesh.reduced_dim(1) + blockSize.y - 1) / blockSize.y,
             (mesh.reduced_dim(2) + blockSize.z - 1) / blockSize.z);

    Kernels::compute_e_update3d<<<gridSize, blockSize>>>(
        get_pitchptr(data.E.data(0)), get_pitchptr(data.E.data(1)),
        get_pitchptr(data.E.data(2)), get_pitchptr(data.B.data(0)),
        get_pitchptr(data.B.data(1)), get_pitchptr(data.B.data(2)),
        get_pitchptr(data.Bbg.data(0)), get_pitchptr(data.Bbg.data(1)),
        get_pitchptr(data.Bbg.data(2)), get_pitchptr(data.J.data(0)),
        get_pitchptr(data.J.data(1)), get_pitchptr(data.J.data(2)), dt);
  }
  CudaCheckError();
}

void
field_solver::apply_outflow_boundary(sim_data &data, double time) {}

void
field_solver::compute_divs(sim_data& data) {
  dim3 blockSize, gridSize;
  auto &grid = m_env.grid();
  auto &mesh = grid.mesh();
  if (grid.dim() == 1) {
    blockSize = dim3(512);
    gridSize =
        dim3((mesh.reduced_dim(0) + blockSize.x - 1) / blockSize.x);
    Kernels::compute_divs_1d<<<gridSize, blockSize>>>(
        get_pitchptr(data.E.data(0)), get_pitchptr(data.E.data(1)),
        get_pitchptr(data.E.data(2)), get_pitchptr(data.B.data(0)),
        get_pitchptr(data.B.data(1)), get_pitchptr(data.B.data(2)),
        get_pitchptr(data.Ebg.data(0)), get_pitchptr(data.Ebg.data(1)),
        get_pitchptr(data.Ebg.data(2)), get_pitchptr(data.Bbg.data(0)),
        get_pitchptr(data.Bbg.data(1)), get_pitchptr(data.Bbg.data(2)),
        get_pitchptr(data.divE), get_pitchptr(data.divB));
    CudaCheckError();
  } else if (grid.dim() == 2) {
    blockSize = dim3(32, 16);
    gridSize =
        dim3((mesh.reduced_dim(0) + blockSize.x - 1) / blockSize.x,
             (mesh.reduced_dim(1) + blockSize.y - 1) / blockSize.y);

    Kernels::compute_divs_2d<<<gridSize, blockSize>>>(
        get_pitchptr(data.E.data(0)), get_pitchptr(data.E.data(1)),
        get_pitchptr(data.E.data(2)), get_pitchptr(data.B.data(0)),
        get_pitchptr(data.B.data(1)), get_pitchptr(data.B.data(2)),
        get_pitchptr(data.Ebg.data(0)), get_pitchptr(data.Ebg.data(1)),
        get_pitchptr(data.Ebg.data(2)), get_pitchptr(data.Bbg.data(0)),
        get_pitchptr(data.Bbg.data(1)), get_pitchptr(data.Bbg.data(2)),
        get_pitchptr(data.divE), get_pitchptr(data.divB));
    CudaCheckError();
  } else if (grid.dim() == 3) {
    blockSize = dim3(32, 8, 4);
    gridSize =
        dim3((mesh.reduced_dim(0) + blockSize.x - 1) / blockSize.x,
             (mesh.reduced_dim(1) + blockSize.y - 1) / blockSize.y,
             (mesh.reduced_dim(2) + blockSize.z - 1) / blockSize.z);

    Kernels::compute_divs_3d<<<gridSize, blockSize>>>(
        get_pitchptr(data.E.data(0)), get_pitchptr(data.E.data(1)),
        get_pitchptr(data.E.data(2)), get_pitchptr(data.B.data(0)),
        get_pitchptr(data.B.data(1)), get_pitchptr(data.B.data(2)),
        get_pitchptr(data.Ebg.data(0)), get_pitchptr(data.Ebg.data(1)),
        get_pitchptr(data.Ebg.data(2)), get_pitchptr(data.Bbg.data(0)),
        get_pitchptr(data.Bbg.data(1)), get_pitchptr(data.Bbg.data(2)),
        get_pitchptr(data.divE), get_pitchptr(data.divB));
    CudaCheckError();
  }
}

}  // namespace Aperture
