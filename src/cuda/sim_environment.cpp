#include "sim_environment_impl.hpp"
#include "cuda/constant_mem_func.h"
#include <cuda_runtime.h>

namespace Aperture {

void
sim_environment::send_array_x(multi_array<Scalar>& array, int dir) {
  int dest, origin;
  MPI_Status status;

  dest = (dir == -1 ? m_domain_info.neighbor_left[0] : m_domain_info.neighbor_right[0]);
  origin = (dir == -1 ? m_domain_info.neighbor_right[0] : m_domain_info.neighbor_left[0]);

  auto& mesh = m_grid->mesh();

  cudaExtent ext = make_cudaExtent(mesh.guard[0] * sizeof(Scalar),
                                   mesh.dims[1], mesh.dims[2]);

  cudaMemcpy3DParms copy_parms = {0};
  copy_parms.srcPtr = make_cudaPitchedPtr(
      array.dev_ptr(), mesh.dims[0] * sizeof(Scalar), mesh.dims[0],
      mesh.dims[1]);
  copy_parms.srcPos =
      make_cudaPos((dir == -1 ? mesh.guard[0]
                              : mesh.dims[0] - 2 * mesh.guard[0]) *
                       sizeof(Scalar),
                   0, 0);
  copy_parms.dstPtr = make_cudaPitchedPtr(
      m_send_buffers[0].dev_ptr(), mesh.guard[0] * sizeof(Scalar),
      mesh.guard[0], mesh.dims[1]);
  copy_parms.dstPos = make_cudaPos(0, 0, 0);
  copy_parms.extent = ext;
  copy_parms.kind = cudaMemcpyDeviceToDevice;
  cudaMemcpy3D(&copy_parms);

  MPI_Sendrecv(m_send_buffers[0].dev_ptr(), m_send_buffers[0].size(),
               m_scalar_type, dest, 0, m_recv_buffers[0].dev_ptr(),
               m_recv_buffers[0].size(), m_scalar_type, origin, 0,
               m_cart, &status);

  if (status.MPI_SOURCE != MPI_PROC_NULL) {
    copy_parms.srcPtr = make_cudaPitchedPtr(
        m_recv_buffers[0].dev_ptr(), mesh.guard[0] * sizeof(Scalar),
        mesh.guard[0], mesh.dims[1]);
    copy_parms.srcPos = make_cudaPos(0, 0, 0);
    copy_parms.dstPtr = make_cudaPitchedPtr(
        array.dev_ptr(), mesh.dims[0] * sizeof(Scalar),
        mesh.dims[0], mesh.dims[1]);
    copy_parms.dstPos = make_cudaPos(
        (dir == -1 ? mesh.dims[0] - mesh.guard[0] : 0) *
            sizeof(Scalar),
        0, 0);
    copy_parms.extent = ext;
    copy_parms.kind = cudaMemcpyDeviceToDevice;
    cudaMemcpy3D(&copy_parms);
  }
}

void
sim_environment::send_array_y(multi_array<Scalar>& array, int dir) {
  int dest, origin;
  MPI_Status status;

  dest = (dir == -1 ? m_domain_info.neighbor_left[1] : m_domain_info.neighbor_right[1]);
  origin = (dir == -1 ? m_domain_info.neighbor_right[1] : m_domain_info.neighbor_left[1]);

  auto& mesh = m_grid->mesh();

  // array.copy_to_y_buffer(m_send_buffers[1], m_grid.guard[1], dir);
  cudaExtent ext =
      make_cudaExtent(mesh.dims[0] * sizeof(Scalar), mesh.guard[1], mesh.dims[2]);

  cudaMemcpy3DParms copy_parms = {0};
  copy_parms.srcPtr = make_cudaPitchedPtr(
      array.dev_ptr(), mesh.dims[0] * sizeof(Scalar), mesh.dims[0],
      mesh.dims[1]);
  copy_parms.srcPos =
      make_cudaPos(0,
                   (dir == -1 ? mesh.guard[1]
                              : mesh.dims[1] - 2 * mesh.guard[1]),
                   0);
  copy_parms.dstPtr = make_cudaPitchedPtr(
      m_send_buffers[1].dev_ptr(), mesh.dims[0] * sizeof(Scalar),
      mesh.dims[0], mesh.guard[1]);
  copy_parms.dstPos = make_cudaPos(0, 0, 0);
  copy_parms.extent = ext;
  copy_parms.kind = cudaMemcpyDeviceToDevice;
  cudaMemcpy3D(&copy_parms);

  MPI_Sendrecv(m_send_buffers[1].dev_ptr(), m_send_buffers[1].size(),
               m_scalar_type, dest, 0, m_recv_buffers[1].dev_ptr(),
               m_recv_buffers[1].size(), m_scalar_type, origin, 0,
               m_cart, &status);

  if (status.MPI_SOURCE != MPI_PROC_NULL) {
    copy_parms.srcPtr = make_cudaPitchedPtr(
        m_recv_buffers[1].dev_ptr(), mesh.dims[0] * sizeof(Scalar),
        mesh.dims[0], mesh.guard[1]);
    copy_parms.srcPos = make_cudaPos(0, 0, 0);
    copy_parms.dstPtr = make_cudaPitchedPtr(
        array.dev_ptr(), mesh.dims[0] * sizeof(Scalar),
        mesh.dims[0], mesh.dims[1]);
    copy_parms.dstPos = make_cudaPos(
        0, (dir == -1 ? mesh.dims[1] - mesh.guard[1] : 0), 0);
    copy_parms.extent = ext;
    copy_parms.kind = cudaMemcpyDeviceToDevice;
    cudaMemcpy3D(&copy_parms);
  }
}

void
sim_environment::setup_env_extra() {
  // Poll the system to detect how many GPUs are on the node
  int n_devices;
  cudaGetDeviceCount(&n_devices);
  if (n_devices <= 0) {
    Logger::err("No usable Cuda device found!!");
    exit(1);
  }
  m_dev_id = m_domain_info.rank % n_devices;

  // m_dev_map.resize(n_devices);
  Logger::print_info("Found {} Cuda devices, using dev {}", n_devices,
                     m_dev_id);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, m_dev_id);
  Logger::print_info("    Device Number: {}", m_dev_id);
  Logger::print_info("    Device Name: {}", prop.name);
  Logger::print_info("    Device Total Memory: {}MiB",
                     prop.totalGlobalMem / (1024 * 1024));

  init_dev_params(m_params);
  init_dev_mesh(m_grid->mesh());

  init_dev_charges(m_charges.data());
  init_dev_masses(m_masses.data());
}


}
