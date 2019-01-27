#include "domain_communicator.h"
#include "core/detail/multi_array_iter_impl.hpp"
#include "core/detail/multi_array_utils.hpp"

#define INSTANTIATE_FUNCTIONS(type)                                  \
  template void DomainCommunicator::get_guard_cells_leftright<type>( \
      int dir, cu_multi_array<type>& array, CommTags leftright,          \
      const Grid& grid);                                             \
  template void DomainCommunicator::put_guard_cells_leftright<type>( \
      int dir, cu_multi_array<type>& array, CommTags leftright,          \
      const Grid& grid, int component);                              \
  template void DomainCommunicator::get_guard_cells<type>(           \
      cu_multi_array<type> & array, const Grid& grid);                   \
  template void DomainCommunicator::put_guard_cells<type>(           \
      cu_multi_array<type> & array, const Grid& grid, int component)

using namespace Aperture;

const int max_buff_num = 10000;

DomainCommunicator::DomainCommunicator(cu_sim_environment& env) : m_env(env) {
  for (int i = 0; i < NUM_PTC_BUFFERS; i++) {
    m_ptc_buffers[i].resize(max_buff_num, single_particle_t());
    m_photon_buffers[i].resize(max_buff_num, single_photon_t());
  }

  // TODO: Fix magic number!!
  m_ptc_partition.resize(27);

  // resize the field buffer arrays
  for (unsigned int i = 0; i < env.local_grid().dim(); i++) {
    Extent ext = env.local_grid().mesh().extent();
    ext[i] = env.local_grid().mesh().guard[i];

    m_field_buf_send[i].resize(ext);
    m_field_buf_recv[i].resize(ext);
  }
}

DomainCommunicator::~DomainCommunicator() {}

template <typename T>
void
DomainCommunicator::get_guard_cells_leftright(int dir,
                                              cu_multi_array<T>& array,
                                              CommTags leftright,
                                              const Grid& grid) {
  if (dir >= 3 || dir < 0)
    throw std::invalid_argument("Invalid direction!");

  auto& domain = m_env.domain_info();
  if (m_env.cartesian().dim(dir) < 2 && !domain.is_periodic[dir])
    return;
  auto& mesh = grid.mesh();

  MPI_Request request[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
  MPI_Status status[2];

  // Obtain the starting index of send and receive buffers in the grid
  Index sendId = Index(0, 0, 0);
  sendId[dir] = (leftright == CommTags::left ? mesh.guard[dir]
                                             : mesh.reduced_dim(dir));
  Index recvId = Index(0, 0, 0);
  recvId[dir] =
      (leftright == CommTags::left ? mesh.dims[dir] - mesh.guard[dir]
                                   : 0);
  Extent sendExt = Extent(mesh.dims[0], mesh.dims[1], mesh.dims[2]);
  sendExt[dir] = mesh.guard[dir];
  // if (dir == 0) {
  //   sendId = Index(
  //       (leftright == CommTags::left ? mesh.guard[0] :
  //       mesh.reducedDim(0)), 0, 0);
  //   recvId = Index(
  //       (leftright == CommTags::left ? mesh.dims[0] - mesh.guard[0] :
  //       0), 0, 0);
  //   sendExt = Extent(mesh.guard[0], mesh.dims[1], mesh.dims[2]);
  // } else if (dir == 1) {
  //   sendId = Index(
  //       0, (leftright == CommTags::left ? mesh.guard[1] :
  //       mesh.reducedDim(1)), 0);
  //   recvId = Index(
  //       0, (leftright == CommTags::left ? mesh.dims[1] -
  //       mesh.guard[1] : 0), 0);
  //   sendExt = Extent(mesh.dims[0], mesh.guard[1], mesh.dims[2]);
  // } else if (dir == 2) {
  //   sendId = Index(
  //       0, 0, (leftright == CommTags::left ? mesh.guard[2] :
  //       mesh.reducedDim(2)));
  //   recvId = Index(
  //       0, 0, (leftright == CommTags::left ? mesh.dims[2] -
  //       mesh.guard[2] : 0));
  //   sendExt = Extent(mesh.dims[0], mesh.dims[1], mesh.guard[2]);
  // }

  // Determine the from and destination rank
  int rank_from =
      (leftright == CommTags::left ? domain.cart_neighbor_right[dir]
                                   : domain.cart_neighbor_left[dir]);
  int rank_dest =
      (leftright == CommTags::left ? domain.cart_neighbor_left[dir]
                                   : domain.cart_neighbor_right[dir]);

  if (rank_dest != NEIGHBOR_NULL) {
    // Copy the content to the send buffer
    copy_to_linear(m_field_buf_send[dir].begin(), array.index(sendId),
                   sendExt);
    m_env.cartesian().Isend(rank_dest, (int)leftright,
                            m_field_buf_send[dir].data(),
                            sendExt.size(), request[0]);
  }

  if (rank_from != NEIGHBOR_NULL) {
    m_env.cartesian().Irecv(rank_from, (int)leftright,
                            m_field_buf_recv[dir].data(),
                            sendExt.size(), request[1]);
  }

  // wait before add_from_linear
  m_env.cartesian().waitall(2, request, status);

  if (rank_from != NEIGHBOR_NULL) {
    copy_from_linear(array.index(recvId), m_field_buf_recv[dir].begin(),
                     sendExt);
  }
}

template <typename T>
void
DomainCommunicator::put_guard_cells_leftright(int dir,
                                              cu_multi_array<T>& array,
                                              CommTags leftright,
                                              const Grid& grid,
                                              int stagger) {
  if (dir >= 3 || dir < 0)
    throw std::invalid_argument("Invalid direction!");

  auto& domain = m_env.domain_info();
  if (m_env.cartesian().dim(dir) < 2 && !domain.is_periodic[dir])
    return;
  auto& mesh = grid.mesh();

  MPI_Request request[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
  MPI_Status status[2];

  // Obtain the starting index of send and receive buffers in the grid
  Index sendId = Index(0, 0, 0);
  // TODO: Is this right? overall recess by 1 unit
  sendId[dir] = (leftright == CommTags::left
                     ? 0
                     : mesh.dims[dir] - mesh.guard[dir] - stagger);
  Index recvId = Index(0, 0, 0);
  recvId[dir] =
      (leftright == CommTags::left ? mesh.reduced_dim(dir)
                                   : mesh.guard[dir] - stagger);
  Extent sendExt = Extent(mesh.dims[0], mesh.dims[1], mesh.dims[2]);
  sendExt[dir] = mesh.guard[dir];
  // if (dir == 0) {
  //   sendId = Index(
  //       (leftright == CommTags::left ? mesh.guard[0] :
  //       mesh.reducedDim(0)), 0, 0);
  //   recvId = Index(
  //       (leftright == CommTags::left ? mesh.dims[0] - mesh.guard[0] :
  //       0), 0, 0);
  //   sendExt = Extent(mesh.guard[0], mesh.dims[1], mesh.dims[2]);
  // } else if (dir == 1) {
  //   sendId = Index(
  //       0, (leftright == CommTags::left ? mesh.guard[1] :
  //       mesh.reducedDim(1)), 0);
  //   recvId = Index(
  //       0, (leftright == CommTags::left ? mesh.dims[1] -
  //       mesh.guard[1] : 0), 0);
  //   sendExt = Extent(mesh.dims[0], mesh.guard[1], mesh.dims[2]);
  // } else if (dir == 2) {
  //   sendId = Index(
  //       0, 0, (leftright == CommTags::left ? mesh.guard[2] :
  //       mesh.reducedDim(2)));
  //   recvId = Index(
  //       0, 0, (leftright == CommTags::left ? mesh.dims[2] -
  //       mesh.guard[2] : 0));
  //   sendExt = Extent(mesh.dims[0], mesh.dims[1], mesh.guard[2]);
  // }

  // Determine the from and destination rank
  int rank_from =
      (leftright == CommTags::left ? domain.cart_neighbor_right[dir]
                                   : domain.cart_neighbor_left[dir]);
  int rank_dest =
      (leftright == CommTags::left ? domain.cart_neighbor_left[dir]
                                   : domain.cart_neighbor_right[dir]);

  if (rank_dest != NEIGHBOR_NULL) {
    // Copy the content to the send buffer
    copy_to_linear(m_field_buf_send[dir].begin(), array.index(sendId),
                   sendExt);
    m_env.cartesian().Isend(rank_dest, (int)leftright,
                            m_field_buf_send[dir].data(),
                            sendExt.size(), request[0]);
  }

  if (rank_from != NEIGHBOR_NULL) {
    m_env.cartesian().Irecv(rank_from, (int)leftright,
                            m_field_buf_recv[dir].data(),
                            sendExt.size(), request[1]);
  }

  // wait before add_from_linear
  m_env.cartesian().waitall(2, request, status);

  if (rank_from != NEIGHBOR_NULL) {
    add_from_linear(array.index(recvId), m_field_buf_recv[dir].begin(),
                    sendExt);
  }
}

template <typename T>
void
DomainCommunicator::get_guard_cells(cu_multi_array<T>& array,
                                    const Aperture::Grid& grid) {
  for (unsigned int i = 0; i < grid.dim(); i++) {
    get_guard_cells_leftright(i, array, CommTags::left, grid);
    get_guard_cells_leftright(i, array, CommTags::right, grid);
  }
}

template <typename T>
void
DomainCommunicator::put_guard_cells(cu_multi_array<T>& array,
                                    const Aperture::Grid& grid,
                                    int stagger) {
  for (unsigned int i = 0; i < grid.dim(); i++) {
    put_guard_cells_leftright(i, array, CommTags::left, grid, stagger);
    put_guard_cells_leftright(i, array, CommTags::right, grid, stagger);
  }
}

void
DomainCommunicator::get_guard_cells(vec_field_t& field) {
  for (int i = 0; i < 3; i++) {
    get_guard_cells(field.data(i), field.grid());
  }
}

void
DomainCommunicator::get_guard_cells(sca_field_t& field) {
  get_guard_cells(field.data(), field.grid());
}

void
DomainCommunicator::put_guard_cells(vec_field_t& field) {
  for (int i = 0; i < 3; i++) {
    put_guard_cells(field.data(i), field.grid(), field.stagger(i)[i]);
  }
}

void
DomainCommunicator::put_guard_cells(sca_field_t& field) {
  // Always assume the stagger of scalar field is zero in all
  // directions. Is this good enough?
  put_guard_cells(field.data(), field.grid(), 0);
}

template <typename ParticleClass>
void
DomainCommunicator::send_recv_particles(
    particle_base<ParticleClass>& particles,
    const Aperture::Grid& grid) {}

INSTANTIATE_FUNCTIONS(double);
INSTANTIATE_FUNCTIONS(float);
INSTANTIATE_FUNCTIONS(int);
INSTANTIATE_FUNCTIONS(unsigned int);
INSTANTIATE_FUNCTIONS(char);

template void DomainCommunicator::send_recv_particles<
    single_particle_t>(particle_base<single_particle_t>& particles,
                       const Aperture::Grid& grid);
template void DomainCommunicator::send_recv_particles<single_photon_t>(
    particle_base<single_photon_t>& particles,
    const Aperture::Grid& grid);
