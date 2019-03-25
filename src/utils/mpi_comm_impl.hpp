#ifndef _MPI_COMM_IMPL_H_
#define _MPI_COMM_IMPL_H_

#include "core/enum_types.h"
#include "utils/logger.h"
#include "utils/mpi_comm.h"
#include <stddef.h>
#include <stdexcept>
#include <unordered_map>

namespace Aperture {

MPICommBase::MPICommBase() {}

MPICommBase::MPICommBase(MPI_Comm comm) {
  _comm = comm;
  MPI_Comm_rank(_comm, &_rank);
  MPI_Comm_size(_comm, &_size);
}

MPICommBase::~MPICommBase() {}

void
MPICommBase::print_rank() const {
  std::cout << "Rank of this processor is " << _rank << " out of "
            << _size << std::endl;
}

void
MPICommBase::barrier() const {
  MPI_Barrier(_comm);
}

template <typename T>
void
MPICommBase::send(int dest_rank, int tag, const T* values,
                  int n) const {
  MPI_Datatype type = MPI_Helper::get_mpi_datatype(*values);

  int error_code =
      MPI_Send((void*)values, n, type, dest_rank, tag, _comm);
  MPI_Helper::handle_mpi_error(error_code, _rank);
}

template <typename T>
void
MPICommBase::send(int dest_rank, int tag, const T& value) const {
  MPI_Datatype type = MPI_Helper::get_mpi_datatype(value);

  int error_code =
      MPI_Send((void*)&value, 1, type, dest_rank, tag, _comm);

  MPI_Helper::handle_mpi_error(error_code, _rank);
}

// void MPICommBase::send(int dest_rank, int tag) const {
//   char byte = 0;

//   int error_code = MPI_Send((void*)&byte, 0, MPI_BYTE, dest_rank,
//   tag, _comm);

//   MPI_Helper::handle_mpi_error(error_code, _rank);
// }

template <typename T>
void
MPICommBase::Isend(int dest_rank, int tag, const T* values, int n,
                   MPI_Request& request) const {
  MPI_Datatype type = MPI_Helper::get_mpi_datatype(*values);

  int error_code = MPI_Isend((void*)values, n, type, dest_rank, tag,
                             _comm, &request);
  MPI_Helper::handle_mpi_error(error_code, _rank);
}

template <typename T>
void
MPICommBase::recv(int source_rank, int tag, T* values, int n,
                  MPI_Status& status) const {
  MPI_Datatype type = MPI_Helper::get_mpi_datatype(*values);

  int error_code = MPI_Recv((void*)values, n, type, source_rank, tag,
                            _comm, &status);

  MPI_Helper::handle_mpi_error(error_code, _rank);
}

template <typename T>
void
MPICommBase::recv(int source_rank, int tag, T* values, int n) const {
  MPI_Datatype type = MPI_Helper::get_mpi_datatype(*values);

  int error_code = MPI_Recv((void*)values, n, type, source_rank, tag,
                            _comm, MPI_STATUS_IGNORE);

  MPI_Helper::handle_mpi_error(error_code, _rank);
}

template <typename T>
void
MPICommBase::recv(int source_rank, int tag, T& value) const {
  MPI_Datatype type = MPI_Helper::get_mpi_datatype(value);

  int error_code = MPI_Recv((void*)&value, 1, type, source_rank, tag,
                            _comm, MPI_STATUS_IGNORE);

  MPI_Helper::handle_mpi_error(error_code, _rank);
}

// template <typename T>
// void MPICommBase::recv(int source_rank, int tag) {
//   char byte = 0;
//   int error_code = MPI_Recv((void*)&byte, 0, MPI_BYTE, source_rank,
//   tag, _comm,
//                             MPI_STATUS_IGNORE);

//   MPI_Helper::handle_mpi_error(error_code, _rank);
// }

template <typename T>
void
MPICommBase::Irecv(int source_rank, int tag, T* values, int n,
                   MPI_Request& request) const {
  MPI_Datatype type = MPI_Helper::get_mpi_datatype(*values);

  int error_code = MPI_Irecv((void*)values, n, type, source_rank, tag,
                             _comm, &request);

  MPI_Helper::handle_mpi_error(error_code, _rank);
}

void
MPICommBase::get_recv_count(MPI_Status& status, MPI_Datatype datatype,
                            int& count) const {
  int error_code = MPI_Get_count(&status, datatype, &count);

  MPI_Helper::handle_mpi_error(error_code, _rank);
}

template <typename T>
void
MPICommBase::send_recv(int source_rank, const T* src_values,
                       int dest_rank, T* dest_values, int tag,
                       int n) const {
  MPI_Datatype type = MPI_Helper::get_mpi_datatype(*src_values);

  int error_code = MPI_Sendrecv(
      (void*)src_values, n, type, dest_rank, tag, (void*)dest_values, n,
      type, source_rank, tag, _comm, MPI_STATUS_IGNORE);

  MPI_Helper::handle_mpi_error(error_code, _rank);
}

template <typename T>
void
MPICommBase::send_recv(int source_rank, const T& src_value,
                       int dest_rank, T& dest_value, int tag) const {
  MPI_Datatype type = MPI_Helper::get_mpi_datatype(src_value);

  int error_code = MPI_Sendrecv(
      (void*)&src_value, 1, type, dest_rank, tag, (void*)&dest_value, 1,
      type, source_rank, tag, _comm, MPI_STATUS_IGNORE);

  MPI_Helper::handle_mpi_error(error_code, _rank);
}

template <typename T>
void
MPICommBase::send_recv(int source_rank, int dest_rank, int tag) const {
  char send_buf = 0;
  char recv_buf = 0;

  int error_code = MPI_Sendrecv(
      (void*)&send_buf, 1, MPI_BYTE, dest_rank, tag, (void*)&recv_buf,
      1, MPI_BYTE, source_rank, tag, _comm, MPI_STATUS_IGNORE);

  MPI_Helper::handle_mpi_error(error_code, _rank);
}

template <typename T>
void
MPICommBase::gather(const T* send_buf, int sendcount, T* recv_buf,
                    int recvcount, int root) const {
  MPI_Datatype type = MPI_Helper::get_mpi_datatype(*send_buf);

  int error_code =
      MPI_Gather((void*)send_buf, sendcount, type, (void*)recv_buf,
                 recvcount, type, root, _comm);
  // MPI_Status status[3];

  MPI_Helper::handle_mpi_error(error_code, _rank);
}

template <typename T>
void
MPICommBase::all_gather(const T* send_buf, int sendcount, T* recv_buf,
                        int recvcount) const {
  MPI_Datatype type = MPI_Helper::get_mpi_datatype(*send_buf);

  int error_code =
      MPI_Allgather((void*)send_buf, sendcount, type, (void*)recv_buf,
                    recvcount, type, _comm);
  // MPI_Status status[3];

  MPI_Helper::handle_mpi_error(error_code, _rank);
}

template <typename T>
void
MPICommBase::gather(const T* send_buf, int sendcount, int root) const {
  // on non-root processes, recvbuf, recvcount ,recvdatatype are ignored
  MPI_Datatype type = MPI_Helper::get_mpi_datatype(*send_buf);

  int error_code = MPI_Gather((void*)send_buf, sendcount, type, NULL, 0,
                              MPI_INT, root, _comm);

  MPI_Helper::handle_mpi_error(error_code, _rank);
}

template <typename T>
void
MPICommBase::gather_inplace(T* recv_buf, int recvcount,
                            int root) const {
  // inplace gather ignores sendcount and senddatatype
  MPI_Datatype type = MPI_Helper::get_mpi_datatype(*recv_buf);

  int error_code = MPI_Gather(MPI_IN_PLACE, 0, MPI_INT, (void*)recv_buf,
                              recvcount, type, root, _comm);

  MPI_Helper::handle_mpi_error(error_code, _rank);
}

template <typename T>
void
MPICommBase::gatherv(const T* send_buf, int sendcount, T* recv_buf,
                     int* recvcounts, int* displs, int root) const {
  MPI_Datatype type = MPI_Helper::get_mpi_datatype(*send_buf);

  int error_code =
      MPI_Gatherv((void*)send_buf, sendcount, type, (void*)recv_buf,
                  recvcounts, displs, type, root, _comm);

  MPI_Helper::handle_mpi_error(error_code, _rank);
}

// this version is mostly used by non-root processes becuase in this
// case recv_buf and recvcount are not significant
template <typename T>
void
MPICommBase::gatherv(const T* send_buf, int sendcount, int root) const {
  MPI_Datatype type = MPI_Helper::get_mpi_datatype(*send_buf);

  int error_code = MPI_Gatherv((void*)send_buf, sendcount, type, NULL,
                               NULL, NULL, MPI_INT, root, _comm);

  MPI_Helper::handle_mpi_error(error_code, _rank);
}

// this version is called by root in an in-place manner
template <typename T>
void
MPICommBase::gatherv_inplace(T* recv_buf, int* recvcounts, int* displs,
                             int root) const {
  MPI_Datatype type = MPI_Helper::get_mpi_datatype(*recv_buf);

  int error_code =
      MPI_Gatherv(MPI_IN_PLACE, 0, MPI_INT, (void*)recv_buf, recvcounts,
                  displs, type, root, _comm);

  MPI_Helper::handle_mpi_error(error_code, _rank);
}

void
MPICommBase::waitall(int length_of_array,
                     MPI_Request* array_of_requests,
                     MPI_Status* array_of_statuses) const {
  int error_code = MPI_Waitall(length_of_array, array_of_requests,
                               array_of_statuses);

  MPI_Helper::handle_mpi_error(error_code, _rank);
}

MPI_Status
MPICommBase::probe(int source, int tag) const {
  MPI_Status status;

  int error_code = MPI_Probe(source, tag, _comm, &status);
  MPI_Helper::handle_mpi_error(error_code, _rank);

  return status;
}

template <typename T>
int
MPICommBase::get_count(const T* value, MPI_Status* status) const {
  int count = 0;
  MPI_Datatype type = MPI_Helper::get_mpi_datatype(*value);
  MPI_Get_count(status, type, &count);
  return count;
}

MPICommWorld::MPICommWorld() {
  _comm = MPI_COMM_WORLD;
  _name = std::string("Comm world");
  MPI_Comm_rank(_comm, &_rank);
  MPI_Comm_size(_comm, &_size);
}

MPICommCartesian::MPICommCartesian() {
  // initialization of _rank and _size is deferred to createCart
  _name = std::string("Comm cartesian");
}

MPICommCartesian::~MPICommCartesian() {
  if (_ndims > 0) {
    delete[] _dims;
    delete[] _periodic;
    delete[] _coords;
    delete[] _neighbor_right;
    delete[] _neighbor_left;
    delete[] _neighbor_corner;
    delete[] _rows;
    _dims = nullptr;
    _periodic = nullptr;
    _coords = nullptr;
    _neighbor_left = nullptr;
    _neighbor_right = nullptr;
    _neighbor_corner = nullptr;
    _rows = nullptr;
  }
}

void
MPICommCartesian::create_dims(int num_nodes, int ndims, int* dims) {
  int error_code = MPI_Dims_create(num_nodes, ndims, dims);

  MPI_Helper::handle_mpi_error(error_code, _rank);
}

void
MPICommCartesian::create_cart(int ndims, int dims[], bool periodic[],
                              std::vector<int>& ranks) {
  // this means the calling proc is not selected to createCart
  if (ranks.size() == 0) {
    return;
  }

  _ndims = ndims;
  _dims = new int[ndims];
  _periodic = new bool[ndims];
  _coords = new int[ndims];
  _neighbor_right = new int[ndims];
  _neighbor_left = new int[ndims];
  _neighbor_corner = new int[(1 << ndims)];
  int* is_periodic = new int[ndims];

  int cart_size = 1;
  for (int i = 0; i < _ndims; i++) {
    _dims[i] = dims[i];
    _periodic[i] = periodic[i];
    is_periodic[i] = (int)periodic[i];
    _neighbor_left[i] = 0;
    _neighbor_right[i] = 0;
    cart_size *= _dims[i];
  }
  // Logger::print(1, "Cartesian topology created with cart_size =",
  // cart_size, "and total size =", _size);

  int world_size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  if (cart_size > world_size) {
    delete[] is_periodic;
    throw std::invalid_argument(
        "Size of the Cartesian grid exceeds the size of world!");
  }

  //---------- Pick some primary nodes to form the Cartesian group
  //------------
  MPI_Group grp_cart;
  MPI_Group grp_world;
  MPI_Comm_group(MPI_COMM_WORLD, &grp_world);

  MPI_Group_incl(grp_world, ranks.size(), ranks.data(), &grp_cart);
  MPI_Group_free(&grp_world);

  // FIXME: delete b/c no longer need to check this
  // // determine whether calling processor is a primary
  // int rank_cart = MPI_UNDEFINED;
  // MPI_Group_rank(grp_cart, &rank_cart);
  // bool isPrimary = ( rank_cart != MPI_UNDEFINED );

  // if (!isPrimary) {
  //   return;
  // }

  // create the Cartesian communicator
  MPI_Comm comm_tmp;
  // MPI_Comm_create_group must be called by all processors in grp_cart
  MPI_Comm_create_group(MPI_COMM_WORLD, grp_cart, 0, &comm_tmp);

  int error_code =
      MPI_Cart_create(comm_tmp, _ndims, _dims, is_periodic, 1, &_comm);
  MPI_Helper::handle_mpi_error(error_code, _rank);

  MPI_Comm_rank(_comm, &_rank);  // need to do this after
                                 // MPI_Cart_create if reorder is true
  MPI_Comm_size(_comm, &_size);

  MPI_Comm_free(&comm_tmp);  // comm_tmp is a different communicator
                             // than _comm, and is no longer needed

  MPI_Cart_coords(_comm, _rank, _ndims, _coords);

  int* coords_corner = new int[_ndims];

  // Compute the world coordinate of left and right neighbors
  for (int i = 0; i < _ndims; i++) {
    int* coords_right = new int[_ndims];
    int* coords_left = new int[_ndims];
    for (int j = 0; j < _ndims; j++) {
      coords_right[j] = _coords[j] + (i == j);
      coords_left[j] = _coords[j] - (i == j);
    }
    // Logger::print_debug_all("On rank {}, right neighbor in dir {} is
    // {}, {}", _rank, i, coords_right[0], coords_right[1]);
    // Logger::print_debug_all("On rank {}, left neighbor in dir {} is
    // {}, {}", _rank, i, coords_left[0], coords_left[1]);
    if (coords_right[i] >= _dims[i]) {
      if (periodic[i]) {
        coords_right[i] = 0;
        MPI_Cart_rank(_comm, coords_right, &_neighbor_right[i]);
      } else {
        _neighbor_right[i] = NEIGHBOR_NULL;
      }
    } else {
      MPI_Cart_rank(_comm, coords_right, &_neighbor_right[i]);
    }
    if (coords_left[i] < 0) {
      if (periodic[i]) {
        coords_left[i] = _dims[i] - 1;
        MPI_Cart_rank(_comm, coords_left, &_neighbor_left[i]);
      } else {
        _neighbor_left[i] = NEIGHBOR_NULL;
      }
    } else {
      MPI_Cart_rank(_comm, coords_left, &_neighbor_left[i]);
    }
    delete[] coords_right;
    delete[] coords_left;
    // Logger::print_debug_all("On rank {}, right neighbor in dir {} is
    // ({})", _rank, i, _neighbor_right[i]); Logger::print_debug_all("On
    // rank {}, left neighbor in dir {} is ({})", _rank, i,
    // _neighbor_left[i]);
  }

  // Compute the world coordinate of corner neighbors
  for (int i = 0; i < (1 << _ndims); i++) {
    for (int j = 0; j < _ndims; j++) {
      // i is the index of the corner, j is the direction
      // (i & (j + 1)) is 0 for i = 0,1 and 1 for i = 2,3 when j = 2
      // (i & (j + 1)) is 1 for i = 1,3 and 0 for i = 0,2 when j = 1
      coords_corner[j] = _coords[j] + ((i & (j + 1)) == 0 ? -1 : 1);
    }
    bool inGrid = true;
    for (int j = 0; j < _ndims; j++) {
      if (coords_corner[j] >= _dims[j] || coords_corner[j] < 0) {
        inGrid = false;
      }
    }
    if (inGrid)
      MPI_Cart_rank(_comm, coords_corner, &_neighbor_corner[i]);
    else
      _neighbor_corner[i] = NEIGHBOR_NULL;
  }

  // Make group communicators for individual rows in every direction
  // TODO: Check this is correct
  _rows = new MPI_Comm[_ndims];

  int* remain_dims = new int[_ndims];
  for (int i = 0; i < _ndims; ++i) {
    for (int j = 0; j < _ndims; ++j) {
      if (j == i)
        remain_dims[j] = 1;
      else
        remain_dims[j] = 0;
    }
    MPI_Cart_sub(_comm, remain_dims, &_rows[i]);
  }
  //    int remain_dims1[2] = { 1, 0 };
  //    int remain_dims2[2] = { 0, 1 };
  //    MPI_Cart_sub(_comm, remain_dims1, &_col);
  //    MPI_Cart_sub(_comm, remain_dims2, &_row);
  Logger::print_detail(
      "Cartesian topology created with cart_size = {} and total size = "
      "{}",
      cart_size, _size);

  delete[] is_periodic;
  delete[] coords_corner;
  delete[] remain_dims;
}

void
MPICommCartesian::printCoord() const {
  // std::cout << "Rank of this processor is " << _rank << " out of " <<
  // _size
  //           << std::endl;
  std::cout << "Coord of this processor is at (";
  for (int i = 0; i < _ndims; i++) {
    std::cout << _coords[i];
    if (i < _ndims - 1) std::cout << ", ";
  }
  std::cout << ")" << std::endl;

  // MPI_Cart_coords(comm_, rank_, ndims_, coords);
  // std::cout << "This processor is at Cartesian coordinate (" <<
  // _coords[0]
  //           << ", " << _coords[1] << ")" << std::endl;
  // std::cout << "my right neighbor in direction 0 has rank "
  //           << _neighbor_right[0] << std::endl;
  // std::cout << "my right neighbor in direction 1 has rank "
  //           << _neighbor_right[1] << std::endl;
  // << coords_[0] << ", " << coords_[1] << ")" << std::endl;
}

template <typename T>
void
MPICommCartesian::scan(const T* send_buf, T* result_buf, int num,
                       int scan_dir, bool exclusive) const {
  MPI_Datatype type = MPI_Helper::get_mpi_datatype(*send_buf);
  MPI_Comm group = _rows[scan_dir];

  int error_code;
  if (!exclusive)
    error_code = MPI_Scan((void*)send_buf, (void*)result_buf, num, type,
                          MPI_SUM, group);
  else
    error_code = MPI_Exscan((void*)send_buf, (void*)result_buf, num,
                            type, MPI_SUM, group);

  MPI_Helper::handle_mpi_error(error_code, _rank);
}

MPIComm::MPIComm() : MPIComm(nullptr, nullptr) {}

MPIComm::MPIComm(int* argc, char*** argv) {
  int is_initialized = 0;
  MPI_Initialized(&is_initialized);

  if (!is_initialized) {
    if (argc == nullptr && argv == nullptr) {
      MPI_Init(NULL, NULL);
    } else {
      MPI_Init(argc, argv);
    }
  }

  MPI_Helper::register_types();

  _world = std::make_unique<MPICommWorld>();
  _cartesian = std::make_unique<MPICommCartesian>();

  // std::cout << _world->rank() << std::endl;
}

MPIComm::~MPIComm() {
  MPI_Helper::free_types();

  int is_finalized = 0;
  MPI_Finalized(&is_finalized);

  if (!is_finalized) MPI_Finalize();
}

std::vector<int>
MPIComm::get_cartesian_members(int cart_size) {
  std::vector<int> result;
  // if (_world->rank() == _world_root) {
  if (is_world_root()) {
    std::vector<int> cart_members(cart_size);
    // simply use first few members in result in creating cartesian
    for (int i = 0; i < cart_size; ++i) cart_members[i] = i;

    // communicate to all processes by sending nothing to non-primary
    // members
    auto requests = MPI_Helper::null_requests(_world->size());
    for (int i = 0; i < _world->size(); ++i) {
      if (_world_root == i) continue;
      int send_num = (i < cart_size) ? cart_members.size() : 0;
      int tag = i;
      _world->Isend(i, tag, cart_members.data(), send_num, requests[i]);
    }
    // MPI_Helper::waitall( requests.size(), requests.data(),
    // MPI_STATUSES_IGNORE );
    _world->waitall(requests.size(), requests.data(),
                    MPI_STATUSES_IGNORE);

    if (_world_root < cart_size) result = cart_members;

  } else {
    // receive
    int tag = _world->rank();
    MPI_Status status = _world->probe(_world_root, tag);
    int count = _world->get_count(result.data(), &status);
    result.resize(count);
    _world->recv(_world_root, tag, result.data(), result.size());
  }

  return result;
}

}  // namespace Aperture

#endif  // _MPI_COMM_IMPL_H_
