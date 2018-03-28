#include "utils/mpi_comm_impl.hpp"
#include "utils/mpi_helper.h"
#include "data/particle_data.h"
#include <type_traits>
#include <cstddef>

using namespace Aperture;

#define INSTANTIATE_SEND(type)                                          \
  template void MPICommBase::send<type>(int dest_rank, int tag, const type* values, int n) const; \
  template void MPICommBase::send<type>(int dest_rank, int tag, const type& value) const; \
  template void MPICommBase::Isend<type>(int dest_rank, int tag, const type* values, int n, MPI_Request& request) const

#define INSTANTIATE_RECV(type)                                          \
  template void MPICommBase::recv<type>(int source_rank, int tag, type* values, int n, MPI_Status& status) const; \
  template void MPICommBase::recv<type>(int source_rank, int tag, type* values, int n) const; \
  template void MPICommBase::recv<type>(int source_rank, int tag, type& value) const; \
  template void MPICommBase::Irecv<type>(int source_rank, int tag, type* values, int n, MPI_Request& request) const

#define INSTANTIATE_SEND_RECV(type)                                     \
  template void MPICommBase::send_recv<type>(int source_rank, const type* src_values, int dest_rank, type* dest_values, int tag, int n) const; \
  template void MPICommBase::send_recv<type>(int source_rank, const type& src_value, int dest_rank, type& dest_value, int tag) const

#define INSTANTIATE_SCAN(type)                                          \
  template void MPICommCartesian::scan<type>(const type* send_buf, type* result_buf, int num, int scan_dir, bool exclusive) const

#define INSTANTIATE_GATHER(type)                                        \
  template void MPICommBase::gather<type>( const type* send_buf, int sendcount, type* recv_buf, int recvcount, int root ) const; \
  template void MPICommBase::all_gather<type>(const type* send_buf, int sendcount, type* recv_buf, int recvcount) const; \
  template void MPICommBase::gather<type>(const type *send_buf, int sendcount, int root) const; \
  template void MPICommBase::gather_inplace<type>(type *recv_buf, int recvcount, int root) const; \
  template void MPICommBase::gatherv<type>(const type *send_buf, int sendcount, type *recv_buf, int *recvcounts, int *displs, int root) const; \
  template void MPICommBase::gatherv<type>(const type *send_buf, int sendcount, int root) const; \
  template void MPICommBase::gatherv_inplace<type>(type *recv_buf, int *recvcounts, int *displs, int root) const

////////////////////////////////////////////////////////////////////////////////
///  Instantiating send and recv methods
////////////////////////////////////////////////////////////////////////////////
INSTANTIATE_SEND(char);
INSTANTIATE_SEND(short);
INSTANTIATE_SEND(int);
INSTANTIATE_SEND(long);
INSTANTIATE_SEND(unsigned char);
INSTANTIATE_SEND(unsigned short);
INSTANTIATE_SEND(unsigned int);
INSTANTIATE_SEND(unsigned long);
INSTANTIATE_SEND(float);
INSTANTIATE_SEND(double);
//INSTANTIATE_SEND(long double);
INSTANTIATE_SEND(single_particle_t);
INSTANTIATE_SEND(single_photon_t);
INSTANTIATE_SEND(Vec3<float>);
INSTANTIATE_SEND(Vec3<double>);
// INSTANTIATE_SEND(Vec4<float>);
// INSTANTIATE_SEND(Vec4<double>);

INSTANTIATE_RECV(char);
INSTANTIATE_RECV(short);
INSTANTIATE_RECV(int);
INSTANTIATE_RECV(long);
INSTANTIATE_RECV(unsigned char);
INSTANTIATE_RECV(unsigned short);
INSTANTIATE_RECV(unsigned int);
INSTANTIATE_RECV(unsigned long);
INSTANTIATE_RECV(float);
INSTANTIATE_RECV(double);
//INSTANTIATE_RECV(long double);
INSTANTIATE_RECV(single_particle_t);
INSTANTIATE_RECV(single_photon_t);
INSTANTIATE_RECV(Vec3<float>);
INSTANTIATE_RECV(Vec3<double>);
// INSTANTIATE_RECV(Vec4<float>);
// INSTANTIATE_RECV(Vec4<double>);

INSTANTIATE_SEND_RECV(char);
INSTANTIATE_SEND_RECV(short);
INSTANTIATE_SEND_RECV(int);
INSTANTIATE_SEND_RECV(long);
INSTANTIATE_SEND_RECV(unsigned char);
INSTANTIATE_SEND_RECV(unsigned short);
INSTANTIATE_SEND_RECV(unsigned int);
INSTANTIATE_SEND_RECV(unsigned long);
INSTANTIATE_SEND_RECV(float);
INSTANTIATE_SEND_RECV(double);
//INSTANTIATE_SEND_RECV(long double);

INSTANTIATE_SCAN(char);
INSTANTIATE_SCAN(short);
INSTANTIATE_SCAN(int);
INSTANTIATE_SCAN(long);
INSTANTIATE_SCAN(unsigned char);
INSTANTIATE_SCAN(unsigned short);
INSTANTIATE_SCAN(unsigned int);
INSTANTIATE_SCAN(unsigned long);
INSTANTIATE_SCAN(float);
INSTANTIATE_SCAN(double);
//INSTANTIATE_SCAN(long double);

INSTANTIATE_GATHER(char);
INSTANTIATE_GATHER(short);
INSTANTIATE_GATHER(int);
INSTANTIATE_GATHER(long);
INSTANTIATE_GATHER(unsigned char);
INSTANTIATE_GATHER(unsigned short);
INSTANTIATE_GATHER(unsigned int);
INSTANTIATE_GATHER(unsigned long);
INSTANTIATE_GATHER(float);
INSTANTIATE_GATHER(double);
//INSTANTIATE_GATHER(long double);

// INSTANTIATE_GET_RECV_COUNT(single_particle_t);
// INSTANTIATE_GET_RECV_COUNT(single_photon_t);
