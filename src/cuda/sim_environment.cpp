#include "sim_environment_impl.hpp"
#include "cuda/constant_mem_func.h"
#include <cuda_runtime.h>

namespace Aperture {

void
sim_environment::send_array_guard_cells_x(multi_array<Scalar> &array,
                                          int dir) {
  int dest, origin;
  MPI_Status status;
  auto &mesh = m_grid->mesh();

  dest = (dir == -1 ? m_domain_info.neighbor_left[0]
                    : m_domain_info.neighbor_right[0]);
  origin = (dir == -1 ? m_domain_info.neighbor_right[0]
                      : m_domain_info.neighbor_left[0]);

  m_send_buffers[0].copy_from(
      array,
      Index((dir == -1 ? mesh.guard[0]
                       : mesh.dims[0] - 2 * mesh.guard[0]),
            0, 0),
      Index(0, 0, 0), m_send_buffers[0].extent(),
      (int)cudaMemcpyDeviceToHost);

  MPI_Sendrecv(m_send_buffers[0].host_ptr(), m_send_buffers[0].size(),
               m_scalar_type, dest, 0, m_recv_buffers[0].host_ptr(),
               m_recv_buffers[0].size(), m_scalar_type, origin, 0,
               m_cart, &status);

  if (status.MPI_SOURCE != MPI_PROC_NULL) {
    array.copy_from(
        m_recv_buffers[0], Index(0, 0, 0),
        Index((dir == -1 ? mesh.dims[0] - mesh.guard[0] : 0), 0, 0),
        m_recv_buffers[0].extent(),
        (int)cudaMemcpyHostToDevice);
  }
}

void
sim_environment::send_array_guard_cells_y(multi_array<Scalar> &array,
                                          int dir) {
  int dest, origin;
  MPI_Status status;
  auto &mesh = m_grid->mesh();

  dest = (dir == -1 ? m_domain_info.neighbor_left[1]
                    : m_domain_info.neighbor_right[1]);
  origin = (dir == -1 ? m_domain_info.neighbor_right[1]
                      : m_domain_info.neighbor_left[1]);

  m_send_buffers[1].copy_from(
      array,
      Index(0,
            (dir == -1 ? mesh.guard[1]
                       : mesh.dims[1] - 2 * mesh.guard[1]),
            0),
      Index(0, 0, 0), m_send_buffers[1].extent(),
      (int)cudaMemcpyDeviceToHost);

  MPI_Sendrecv(m_send_buffers[1].host_ptr(), m_send_buffers[1].size(),
               m_scalar_type, dest, 0, m_recv_buffers[1].host_ptr(),
               m_recv_buffers[1].size(), m_scalar_type, origin, 0,
               m_cart, &status);

  if (status.MPI_SOURCE != MPI_PROC_NULL) {
    array.copy_from(
        m_recv_buffers[1], Index(0, 0, 0),
        Index(0, (dir == -1 ? mesh.dims[1] - mesh.guard[1] : 0), 0),
        m_recv_buffers[1].extent(),
        (int)cudaMemcpyHostToDevice);
  }
}

void
sim_environment::send_array_guard_cells_z(multi_array<Scalar> &array,
                                          int dir) {
  int dest, origin;
  MPI_Status status;
  auto &mesh = m_grid->mesh();

  dest = (dir == -1 ? m_domain_info.neighbor_left[2]
                    : m_domain_info.neighbor_right[2]);
  origin = (dir == -1 ? m_domain_info.neighbor_right[2]
                      : m_domain_info.neighbor_left[2]);

  m_send_buffers[2].copy_from(
      array,
      Index(0, 0,
            (dir == -1 ? mesh.guard[2]
                       : mesh.dims[2] - 2 * mesh.guard[2])),
      Index(0, 0, 0), m_send_buffers[2].extent(),
      (int)cudaMemcpyDeviceToHost);

  MPI_Sendrecv(m_send_buffers[2].host_ptr(), m_send_buffers[2].size(),
               m_scalar_type, dest, 0, m_recv_buffers[2].host_ptr(),
               m_recv_buffers[2].size(), m_scalar_type, origin, 0,
               m_cart, &status);

  if (status.MPI_SOURCE != MPI_PROC_NULL) {
    array.copy_from(
        m_recv_buffers[2], Index(0, 0, 0),
        Index(0, 0, (dir == -1 ? mesh.dims[2] - mesh.guard[2] : 0)),
        m_recv_buffers[2].extent(),
        (int)cudaMemcpyHostToDevice);
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
  cudaSetDevice(m_dev_id);

  // m_dev_map.resize(n_devices);
  init_dev_params(m_params);
  init_dev_mesh(m_grid->mesh());

  Logger::print_debug("Charges are {}, {}", m_charges[0], m_charges[1]);
  Logger::print_debug("Masses are {}, {}", m_masses[0], m_masses[1]);
  init_dev_charges(m_charges.data());
  init_dev_masses(m_masses.data());

  init_dev_rank(m_domain_info.rank);
}


}
