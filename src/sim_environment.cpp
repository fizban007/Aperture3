#include "sim_environment_impl.hpp"

namespace Aperture {

void
sim_environment::setup_env_extra() {}

template <typename T>
void
sim_environment::send_array_guard_cells_x(multi_array<T> &array, int dir) {
  int dest, origin;
  MPI_Status status;

  dest = (dir == -1 ? m_domain_info.neighbor_left[0] : m_domain_info.neighbor_right[0]);
  origin = (dir == -1 ? m_domain_info.neighbor_right[0] : m_domain_info.neighbor_left[0]);
  
  MPI_Sendrecv(m_send_buffers[0].host_ptr(), m_send_buffers[0].size(),
               m_scalar_type, dest, 0, m_recv_buffers[0].host_ptr(),
               m_recv_buffers[0].size(), m_scalar_type, origin, 0,
               m_cart, &status);

  if (status.MPI_SOURCE != MPI_PROC_NULL) {
    
  }
}

}
