#include "cuda/constant_mem_func.h"
#include "sim_environment_impl.hpp"
#include <cuda_runtime.h>

#include "mpi-ext.h" /* Needed for CUDA-aware check */

namespace Aperture {

void
sim_environment::send_array_guard_cells_single_dir(
    multi_array<Scalar> &array, int dim, int dir) {
  if (dim < 0 || dim >= m_grid->dim()) return;

  int dest, origin;
  MPI_Status status;
  auto &mesh = m_grid->mesh();

  dest = (dir == -1 ? m_domain_info.neighbor_left[dim]
                    : m_domain_info.neighbor_right[dim]);
  origin = (dir == -1 ? m_domain_info.neighbor_right[dim]
                      : m_domain_info.neighbor_left[dim]);

  Index send_idx(0, 0, 0);
  send_idx[dim] = (dir == -1 ? mesh.guard[dim]
                             : mesh.dims[dim] - 2 * mesh.guard[dim]);

  m_send_buffers[dim].copy_from(array, send_idx, Index(0, 0, 0),
                                m_send_buffers[dim].extent(),
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
                                (int)cudaMemcpyDeviceToDevice
#else
                                (int)cudaMemcpyDeviceToHost
#endif
  );

#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
  auto send_ptr = m_send_buffers[dim].dev_ptr();
  auto recv_ptr = m_recv_buffers[dim].dev_ptr();
#else
  auto send_ptr = m_send_buffers[dim].host_ptr();
  auto recv_ptr = m_recv_buffers[dim].host_ptr();
#endif

  // MPI_Sendrecv(send_ptr, m_send_buffers[dim].size(), m_scalar_type,
  //              dest, 0, recv_ptr, m_recv_buffers[dim].size(),
  //              m_scalar_type, origin, 0, m_cart, &status);

  if (origin != MPI_PROC_NULL) {
    MPI_Recv(recv_ptr, m_recv_buffers[dim].size(), m_scalar_type,
             origin, 0, m_cart, &status);

    Index recv_idx(0, 0, 0);
    recv_idx[dim] = (dir == -1 ? mesh.dims[dim] - mesh.guard[dim] : 0);
    array.copy_from(m_recv_buffers[dim], Index(0, 0, 0), recv_idx,
                    m_recv_buffers[dim].extent(),
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
                    (int)cudaMemcpyDeviceToDevice
#else
                    (int)cudaMemcpyHostToDevice
#endif
    );
  }
  if (dest != MPI_PROC_NULL) {
    MPI_Send(send_ptr, m_send_buffers[dim].size(), m_scalar_type, dest,
             0, m_cart);
  }
}

void
sim_environment::send_add_array_guard_cells_single_dir(
    multi_array<Scalar> &array, int dim, int dir) {
  if (dim < 0 || dim >= m_grid->dim()) return;

  int dest, origin;
  MPI_Status status;
  auto &mesh = m_grid->mesh();

  dest = (dir == -1 ? m_domain_info.neighbor_left[dim]
                    : m_domain_info.neighbor_right[dim]);
  origin = (dir == -1 ? m_domain_info.neighbor_right[dim]
                      : m_domain_info.neighbor_left[dim]);

  Index send_idx(0, 0, 0);
  send_idx[dim] = (dir == -1 ? 0 : mesh.dims[dim] - mesh.guard[dim]);
  m_send_buffers[dim].copy_from(array, send_idx, Index(0, 0, 0),
                                m_send_buffers[dim].extent(),
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
                                (int)cudaMemcpyDeviceToDevice
#else
                                (int)cudaMemcpyDeviceToHost
#endif
  );

#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
  auto send_ptr = m_send_buffers[dim].dev_ptr();
  auto recv_ptr = m_recv_buffers[dim].dev_ptr();
#else
  auto send_ptr = m_send_buffers[dim].host_ptr();
  auto recv_ptr = m_recv_buffers[dim].host_ptr();
#endif

  // MPI_Sendrecv(send_ptr,
  //              m_send_buffers[dim].size(), m_scalar_type, dest, 0,
  //              recv_ptr,
  //              m_recv_buffers[dim].size(), m_scalar_type, origin, 0,
  //              m_cart, &status);

  // if (status.MPI_SOURCE != MPI_PROC_NULL) {
  if (origin != MPI_PROC_NULL) {
    MPI_Recv(recv_ptr, m_recv_buffers[dim].size(), m_scalar_type,
             origin, 0, m_cart, &status);

    Index recv_idx(0, 0, 0);
    recv_idx[dim] = (dir == -1 ? mesh.dims[dim] - 2 * mesh.guard[dim]
                               : mesh.guard[dim]);
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
#else
    m_recv_buffers[dim].copy_to_device();
#endif
    array.add_from(m_recv_buffers[dim], Index(0, 0, 0), recv_idx,
                   m_recv_buffers[dim].extent());
  }
  if (dest != MPI_PROC_NULL) {
    MPI_Send(send_ptr, m_send_buffers[dim].size(), m_scalar_type, dest,
             0, m_cart);
  }
}

void
sim_environment::setup_env_extra() {
  // m_dev_map.resize(n_devices);
  init_dev_params(m_params);
  init_dev_mesh(m_grid->mesh());

  Logger::print_debug("Charges are {}, {}", m_charges[0], m_charges[1]);
  Logger::print_debug("Masses are {}, {}", m_masses[0], m_masses[1]);
  init_dev_charges(m_charges.data());
  init_dev_masses(m_masses.data());

  init_dev_rank(m_domain_info.rank);

  // Initialize a managed array for particle_data and photon_data
  int num_ptc_buffers = m_ptc_buffers.size();
  cudaMallocManaged(&m_ptc_buf_ptrs,
                    num_ptc_buffers * sizeof(particle_data));
  cudaMallocManaged(&m_ph_buf_ptrs,
                    num_ptc_buffers * sizeof(photon_data));

  for (int i = 0; i < num_ptc_buffers; i++) {
    visit_struct::for_each(
        m_ptc_buf_ptrs[i], m_ptc_buffers[i].data(),
        [](const char *name, auto &v1, const auto &v2) { v1 = v2; });
    visit_struct::for_each(
        m_ph_buf_ptrs[i], m_ph_buffers[i].data(),
        [](const char *name, auto &v1, const auto &v2) { v1 = v2; });
  }
}

void
sim_environment::destruct_extra() {
  cudaFree(&m_ptc_buf_ptrs);
  cudaFree(&m_ph_buf_ptrs);
}

void
sim_environment::setup_device() {
  // Poll the system to detect how many GPUs are on the node
  int n_devices;
  cudaGetDeviceCount(&n_devices);
  if (n_devices <= 0) {
    Logger::err("No usable Cuda device found!!");
    exit(1);
  }
  m_dev_id = m_domain_info.rank % n_devices;
  cudaSetDevice(m_dev_id);
}

}  // namespace Aperture
