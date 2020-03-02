#include "sim_environment_impl.hpp"

namespace Aperture {

void
sim_environment::send_array_guard_cells_single_dir(
    multi_array<Scalar> &array, int dim, int dir) {
  if (dim < 0 || dim >= (int)m_grid->dim()) return;

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
                                m_send_buffers[dim].extent());

  MPI_Sendrecv(m_send_buffers[dim].host_ptr(),
               m_send_buffers[dim].size(), m_scalar_type, dest, 0,
               m_recv_buffers[dim].host_ptr(),
               m_recv_buffers[dim].size(), m_scalar_type, origin, 0,
               m_cart, &status);

  if (status.MPI_SOURCE != MPI_PROC_NULL) {
    Index recv_idx(0, 0, 0);
    recv_idx[dim] = (dir == -1 ? mesh.dims[dim] - mesh.guard[dim] : 0);
    array.copy_from(m_recv_buffers[dim], Index(0, 0, 0), recv_idx,
                    m_recv_buffers[dim].extent());
  }
}

void
sim_environment::send_add_array_guard_cells_single_dir(
    multi_array<Scalar> &array, int dim, int dir) {
  if (dim < 0 || dim >= (int)m_grid->dim()) return;

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
                                m_send_buffers[dim].extent());

  MPI_Sendrecv(m_send_buffers[dim].host_ptr(),
               m_send_buffers[dim].size(), m_scalar_type, dest, 0,
               m_recv_buffers[dim].host_ptr(),
               m_recv_buffers[dim].size(), m_scalar_type, origin, 0,
               m_cart, &status);

  if (status.MPI_SOURCE != MPI_PROC_NULL) {
    Index recv_idx(0, 0, 0);
    recv_idx[dim] = (dir == -1 ? mesh.dims[dim] - 2 * mesh.guard[dim]
                               : mesh.guard[dim]);
    array.add_from(m_recv_buffers[dim], Index(0, 0, 0), recv_idx,
                   m_recv_buffers[dim].extent());
  }
}

void
sim_environment::setup_env_extra() {}

void
sim_environment::destruct_extra() {}

void
sim_environment::setup_device() {}

}  // namespace Aperture
