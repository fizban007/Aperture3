#ifndef _MPI_COMM_IMPL_H_
#define _MPI_COMM_IMPL_H_

#include "data/enum_types.h"
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
  std::cout << "Rank of this processor is " << _rank << " out of " << _size
            << std::endl;
}

void
MPICommBase::barrier() const {
  MPI_Barrier(_comm);
}

template <typename T>
void
MPICommBase::send(int dest_rank, int tag, const T* values, int n) const {
  MPI_Datatype type = MPI_Helper::get_mpi_datatype(*values);

  int error_code = MPI_Send((void*)values, n, type, dest_rank, tag, _comm);
  MPI_Helper::handle_mpi_error(error_code, _rank);
}

template <typename T>
void
MPICommBase::send(int dest_rank, int tag, const T& value) const {
  MPI_Datatype type = MPI_Helper::get_mpi_datatype(value);

  int error_code = MPI_Send((void*)&value, 1, type, dest_rank, tag, _comm);

  MPI_Helper::handle_mpi_error(error_code, _rank);
}

// void MPICommBase::send(int dest_rank, int tag) const {
//   char byte = 0;

//   int error_code = MPI_Send((void*)&byte, 0, MPI_BYTE, dest_rank, tag,
//   _comm);

//   MPI_Helper::handle_mpi_error(error_code, _rank);
// }

template <typename T>
void
MPICommBase::Isend(int dest_rank, int tag, const T* values, int n,
                   MPI_Request& request) const {
  MPI_Datatype type = MPI_Helper::get_mpi_datatype(*values);

  int error_code =
      MPI_Isend((void*)values, n, type, dest_rank, tag, _comm, &request);
  MPI_Helper::handle_mpi_error(error_code, _rank);
}

template <typename T>
void
MPICommBase::recv(int source_rank, int tag, T* values, int n,
                  MPI_Status& status) const {
  MPI_Datatype type = MPI_Helper::get_mpi_datatype(*values);

  int error_code =
      MPI_Recv((void*)values, n, type, source_rank, tag, _comm, &status);

  MPI_Helper::handle_mpi_error(error_code, _rank);
}

template <typename T>
void
MPICommBase::recv(int source_rank, int tag, T* values, int n) const {
  MPI_Datatype type = MPI_Helper::get_mpi_datatype(*values);

  int error_code = MPI_Recv((void*)values, n, type, source_rank, tag, _comm,
                            MPI_STATUS_IGNORE);

  MPI_Helper::handle_mpi_error(error_code, _rank);
}

template <typename T>
void
MPICommBase::recv(int source_rank, int tag, T& value) const {
  MPI_Datatype type = MPI_Helper::get_mpi_datatype(value);

  int error_code = MPI_Recv((void*)&value, 1, type, source_rank, tag, _comm,
                            MPI_STATUS_IGNORE);

  MPI_Helper::handle_mpi_error(error_code, _rank);
}

// template <typename T>
// void MPICommBase::recv(int source_rank, int tag) {
//   char byte = 0;
//   int error_code = MPI_Recv((void*)&byte, 0, MPI_BYTE, source_rank, tag,
//   _comm,
//                             MPI_STATUS_IGNORE);

//   MPI_Helper::handle_mpi_error(error_code, _rank);
// }

template <typename T>
void
MPICommBase::Irecv(int source_rank, int tag, T* values, int n,
                   MPI_Request& request) const {
  MPI_Datatype type = MPI_Helper::get_mpi_datatype(*values);

  int error_code =
      MPI_Irecv((void*)values, n, type, source_rank, tag, _comm, &request);

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
MPICommBase::send_recv(int source_rank, const T* src_values, int dest_rank,
                       T* dest_values, int tag, int n) const {
  MPI_Datatype type = MPI_Helper::get_mpi_datatype(*src_values);

  int error_code = MPI_Sendrecv((void*)src_values, n, type, dest_rank, tag,
                                (void*)dest_values, n, type, source_rank, tag,
                                _comm, MPI_STATUS_IGNORE);

  MPI_Helper::handle_mpi_error(error_code, _rank);
}

template <typename T>
void
MPICommBase::send_recv(int source_rank, const T& src_value, int dest_rank,
                       T& dest_value, int tag) const {
  MPI_Datatype type = MPI_Helper::get_mpi_datatype(src_value);

  int error_code = MPI_Sendrecv((void*)&src_value, 1, type, dest_rank, tag,
                                (void*)&dest_value, 1, type, source_rank, tag,
                                _comm, MPI_STATUS_IGNORE);

  MPI_Helper::handle_mpi_error(error_code, _rank);
}

template <typename T>
void
MPICommBase::send_recv(int source_rank, int dest_rank, int tag) const {
  char send_buf = 0;
  char recv_buf = 0;

  int error_code = MPI_Sendrecv((void*)&send_buf, 1, MPI_BYTE, dest_rank, tag,
                                (void*)&recv_buf, 1, MPI_BYTE, source_rank, tag,
                                _comm, MPI_STATUS_IGNORE);

  MPI_Helper::handle_mpi_error(error_code, _rank);
}

template <typename T>
void
MPICommBase::gather(const T* send_buf, int sendcount, T* recv_buf,
                    int recvcount, int root) const {
  MPI_Datatype type = MPI_Helper::get_mpi_datatype(*send_buf);

  int error_code = MPI_Gather((void*)send_buf, sendcount, type, (void*)recv_buf,
                              recvcount, type, root, _comm);
  // MPI_Status status[3];

  MPI_Helper::handle_mpi_error(error_code, _rank);
}

template <typename T>
void
MPICommBase::all_gather(const T* send_buf, int sendcount, T* recv_buf,
                        int recvcount) const {
  MPI_Datatype type = MPI_Helper::get_mpi_datatype(*send_buf);

  int error_code = MPI_Allgather((void*)send_buf, sendcount, type,
                                 (void*)recv_buf, recvcount, type, _comm);
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

  MPI_Helper::handle_mpi_error(error_code, *this);
}

template <typename T>
void
MPICommBase::gather_inplace(T* recv_buf, int recvcount, int root) const {
  // inplace gather ignores sendcount and senddatatype
  MPI_Datatype type = MPI_Helper::get_mpi_datatype(*recv_buf);

  int error_code = MPI_Gather(MPI_IN_PLACE, 0, MPI_INT, (void*)recv_buf,
                              recvcount, type, root, _comm);

  MPI_Helper::handle_mpi_error(error_code, *this);
}

template <typename T>
void
MPICommBase::gatherv(const T* send_buf, int sendcount, T* recv_buf,
                     const int* recvcounts, const int* displs, int root) const {
  MPI_Datatype type = MPI_Helper::get_mpi_datatype(*send_buf);

  int error_code =
      MPI_Gatherv((void*)send_buf, sendcount, type, (void*)recv_buf, recvcounts,
                  displs, type, root, _comm);

  MPI_Helper::handle_mpi_error(error_code, *this);
}

// this version is mostly used by non-root processes becuase in this case
// recv_buf and recvcount are not significant
template <typename T>
void
MPICommBase::gatherv(const T* send_buf, int sendcount, int root) const {
  MPI_Datatype type = MPI_Helper::get_mpi_datatype(*send_buf);

  int error_code = MPI_Gatherv((void*)send_buf, sendcount, type, NULL, NULL,
                               NULL, MPI_INT, root, _comm);

  MPI_Helper::handle_mpi_error(error_code, *this);
}

// this version is called by root in an in-place manner
template <typename T>
void
MPICommBase::gatherv_inplace(T* recv_buf, const int* recvcounts,
                             const int* displs, int root) const {
  MPI_Datatype type = MPI_Helper::get_mpi_datatype(*recv_buf);

  int error_code = MPI_Gatherv(MPI_IN_PLACE, 0, MPI_INT, (void*)recv_buf,
                               recvcounts, displs, type, root, _comm);

  MPI_Helper::handle_mpi_error(error_code, *this);
}

void
MPICommBase::waitall(int length_of_array, MPI_Request* array_of_requests,
                     MPI_Status* array_of_statuses) const {
  int error_code =
      MPI_Waitall(length_of_array, array_of_requests, array_of_statuses);

  MPI_Helper::handle_mpi_error(error_code, _rank);
}

MPI_Status
MPICommBase::probe(int source, int tag) const {
  MPI_Status status;

  int error_code = MPI_Probe(source, tag, _comm, &status);
  MPI_Helper::handle_mpi_error(error_code, *this);

  return status;
}

template <typename T>
int
MPICommBase::get_count(const T* value, const MPI_Status* status) const {
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
                              const std::vector<int>& ranks) {
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
  // Logger::print(1, "Cartesian topology created with cart_size =", cart_size,
  // "and total size =", _size);

  int world_size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  if (cart_size > world_size) {
    delete[] is_periodic;
    throw std::invalid_argument(
        "Size of the Cartesian grid exceeds the size of world!");
  }

  //---------- Pick some primary nodes to form the Cartesian group ------------
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
  MPI_Helper::handle_mpi_error(error_code, *this);

  MPI_Comm_rank(_comm, &_rank);  // need to do this after MPI_Cart_create if reorder is true
  MPI_Comm_size(_comm, &_size);

  MPI_Comm_free(&comm_tmp);  // comm_tmp is a different communicator than _comm,
                             // and is no longer needed

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
    // Logger::print_debug_all("On rank {}, right neighbor in dir {} is {}, {}", _rank, i, coords_right[0], coords_right[1]);
    // Logger::print_debug_all("On rank {}, left neighbor in dir {} is {}, {}", _rank, i, coords_left[0], coords_left[1]);
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
    // Logger::print_debug_all("On rank {}, right neighbor in dir {} is ({})", _rank, i, _neighbor_right[i]);
    // Logger::print_debug_all("On rank {}, left neighbor in dir {} is ({})", _rank, i, _neighbor_left[i]);
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
      "Cartesian topology created with cart_size = {} and total size = {}",
      cart_size, _size);

  delete[] is_periodic;
  delete[] coords_corner;
  delete[] remain_dims;
}

void
MPICommCartesian::printCoord() const {
  // std::cout << "Rank of this processor is " << _rank << " out of " << _size
  //           << std::endl;
  std::cout << "Coord of this processor is at (";
  for (int i = 0; i < _ndims; i++) {
    std::cout << _coords[i];
    if (i < _ndims - 1) std::cout << ", ";
  }
  std::cout << ")" << std::endl;

  // MPI_Cart_coords(comm_, rank_, ndims_, coords);
  // std::cout << "This processor is at Cartesian coordinate (" << _coords[0]
  //           << ", " << _coords[1] << ")" << std::endl;
  // std::cout << "my right neighbor in direction 0 has rank "
  //           << _neighbor_right[0] << std::endl;
  // std::cout << "my right neighbor in direction 1 has rank "
  //           << _neighbor_right[1] << std::endl;
  // << coords_[0] << ", " << coords_[1] << ")" << std::endl;
}

template <typename T>
void
MPICommCartesian::scan(const T* send_buf, T* result_buf, int num, int scan_dir,
                       bool exclusive) const {
  MPI_Datatype type = MPI_Helper::get_mpi_datatype(*send_buf);
  MPI_Comm group = _rows[scan_dir];

  int error_code;
  if (!exclusive)
    error_code =
        MPI_Scan((void*)send_buf, (void*)result_buf, num, type, MPI_SUM, group);
  else
    error_code = MPI_Exscan((void*)send_buf, (void*)result_buf, num, type,
                            MPI_SUM, group);

  MPI_Helper::handle_mpi_error(error_code, _rank);
}

MPICommEnsemble::MPICommEnsemble() {
  // again, _rank and _size initialization is deferred to createEnsemble
  _name = std::string("Comm ensemble");
}

void
MPICommEnsemble::create_ensemble(int label, int root_rank_world,
                                 const std::vector<int>& members) {
  // first nullify old communicator
  nullify();

  MPI_Group grp_world;
  MPI_Comm_group(MPI_COMM_WORLD, &grp_world);
  // create the ensemble comm
  MPI_Group grp_ensemble;
  MPI_Group_incl(grp_world, members.size(), members.data(), &grp_ensemble);

  // NOTE MPI_Comm_create_group must be called by all members in the the
  // grp_ensemble.
  MPI_Comm_create_group(MPI_COMM_WORLD, grp_ensemble, 1, &_comm);
  // get the rank with respect to this comm
  MPI_Comm_rank(_comm, &_rank);
  MPI_Comm_size(_comm, &_size);
  // record the rank of the primary in this comm
  MPI_Group_translate_ranks(grp_world, 1, &root_rank_world, grp_ensemble,
                            &_root_ensemble);
  _label = label;

  MPI_Group_free(&grp_world);
  MPI_Group_free(&grp_ensemble);
}

void
MPICommEnsemble::nullify() {
  if (_comm != MPI_COMM_NULL) {
    MPI_Comm_free(&_comm);
  }
  _comm = MPI_COMM_NULL;
  _label = MPI_UNDEFINED;
  _root_ensemble = MPI_UNDEFINED;
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
  _ensemble = std::make_unique<MPICommEnsemble>();

  // std::cout << _world->rank() << std::endl;
  if (is_world_root()) {
    MPI_Comm_group(MPI_COMM_WORLD, &_idle_procs);
    _rank2label.resize(_world->size());
    // NOTE _label2ranks and _tmp_size_map are initialized in
    // InitRosterAfterCartesian because of lack
    // of information here
  }
}

MPIComm::~MPIComm() {
  if (_idle_procs != MPI_GROUP_EMPTY) {
    MPI_Group_free(&_idle_procs);
  }

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

    // communicate to all processes by sending nothing to non-primary members
    auto requests = MPI_Helper::null_requests(_world->size());
    for (int i = 0; i < _world->size(); ++i) {
      if (_world_root == i) continue;
      int send_num = (i < cart_size) ? cart_members.size() : 0;
      int tag = i;
      _world->Isend(i, tag, cart_members.data(), send_num, requests[i]);
    }
    // MPI_Helper::waitall( requests.size(), requests.data(),
    // MPI_STATUSES_IGNORE );
    _world->waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);

    if (_world_root < cart_size) result = cart_members;

    // update _idle_procs group
    MPI_Group idle_old = _idle_procs;
    MPI_Group_excl(idle_old, cart_members.size(), cart_members.data(),
                   &_idle_procs);
    MPI_Group_free(&idle_old);

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

void
MPIComm::init_trivial_ensemble() {
  // NOTE it is assumed that _world_root is one of the primaries. Only primaries
  // need to do this
  if (_cartesian->is_null()) return;
  // cartesian's coord and dim arrays are of length of the actual dimension of
  // the simulation, so they can be short of 3.
  std::array<int, 3> cart_coord = {0, 0, 0};
  std::array<int, 3> cart_dim = {1, 1, 1};
  for (int i = 0; i < _cartesian->ndims(); ++i) {
    cart_coord[i] = _cartesian->coord(i);
    cart_dim[i] = _cartesian->dim(i);
  }

  int my_label =
      coord_to_ensemble_label(cart_coord[0], cart_coord[1], cart_coord[2],
                              cart_dim[0], cart_dim[1], cart_dim[2]);

  int my_world_rank = _world->rank();
  _ensemble->create_ensemble(my_label, my_world_rank, {my_world_rank});

  // initialize the roster variables on _world_root
  int cart_root = 0;
  auto grp_world = _world->group();
  auto grp_cart = _cartesian->group();
  MPI_Group_translate_ranks(grp_world, 1, &_world_root, grp_cart, &cart_root);

  if (_cartesian->rank() == cart_root) {
    // initialize roster variables on world_root
    _label2ranks.resize(_cartesian->size());

    std::vector<int> label_gather(_cartesian->size());
    _cartesian->gather(&my_label, 1, label_gather.data(), 1, cart_root);

    for (int cart_rank = 0; cart_rank < (int)label_gather.size(); ++cart_rank) {
      int label = label_gather[cart_rank];
      // NOTE need to work with world rank
      int world_rank = 0;
      MPI_Group_translate_ranks(grp_cart, 1, &cart_rank, grp_world,
                                &world_rank);
      _label2ranks[label].push_back(world_rank);
      _rank2label[world_rank] = label;
    }

  } else {
    _cartesian->gather(&my_label, 1, cart_root);
  }

  MPI_Group_free(&grp_world);
  MPI_Group_free(&grp_cart);
}

std::vector<int>
MPIComm::group2ranks(MPI_Group group) const {
  int size = 0;
  MPI_Group_size(group, &size);
  std::vector<int> result(size);
  auto grp_world = _world->group();
  for (int i = 0; i < size; ++i) {
    MPI_Group_translate_ranks(group, 1, &i, grp_world, result.data() + i);
  }
  MPI_Group_free(&grp_world);

  return result;
}

std::vector<int>
MPIComm::idle_ranks() const {
  if (_world->rank() != _world_root)
    throw std::runtime_error("None-world-root tries to call idle_ranks!");
  return group2ranks(_idle_procs);
}

std::vector<int>
MPIComm::active_procs() const {
  if (_world->rank() != _world_root)
    throw std::runtime_error("None-world-root tries to call group_active!");

  auto grp_world = _world->group();
  MPI_Group grp_active;
  MPI_Group_difference(grp_world, _idle_procs, &grp_active);
  MPI_Group_free(&grp_world);
  auto result = group2ranks(grp_active);
  MPI_Group_free(&grp_active);
  return result;
}

EnsAdjustInstruction
MPIComm::get_instruction(size_t num_ptc, size_t target_num_ptc_per_proc) {
  // NOTE this varialbe is only significant on world_root
  std::vector<int> new_size_map;
  if (_world->rank() == _world_root) {
    new_size_map.resize(_cartesian->size());
    // gather num_ptc
    std::unique_ptr<size_t[]> num_ptc_gather(new size_t[_world->size()]);
    num_ptc_gather[_world_root] = num_ptc;
    _world->gather_inplace(num_ptc_gather.get(), 1, _world_root);
    // FIXME: generate the new size map
    // NOTE: pay attention to elements from idles
    // NOTE: what if all num_ptcs are zero
    // NOTE: no matther what, new_size_map cannot have elements less than 1

    // // single ensemble, increase it's size by 1
    // new_size_map[0] = std::min( (int)_label2ranks[0].size() + 1,
    // _world->size() );

    static int count = 0;
    // two ensembles
    switch (count) {
      case 0: {
        new_size_map[0] = 1;
        new_size_map[1] = 1;
        count++;
        break;
      }
      case 1: {
        new_size_map[0] = 1;
        new_size_map[1] = 2;
        // count++;
        break;
      }
      case 2: {
        new_size_map[0] = 2;
        new_size_map[1] = 3;
        count++;
        break;
      }
      case 3: {
        new_size_map[0] = 3;
        new_size_map[1] = 2;
        count++;
        break;
      }
      case 4: {
        new_size_map[0] = 3;
        new_size_map[1] = 2;
        count++;
        break;
      }
      case 5: {
        new_size_map[0] = 3;
        new_size_map[1] = 2;
        count++;
        break;
      }
      default: {
        new_size_map[0] = 1;
        new_size_map[1] = 1;
      }
    }
  }

  else {
    _world->gather(&num_ptc, 1, _world_root);
  }

  return decode_raw_instruction(get_raw_instruction(new_size_map));
}

EnsAdjustInstruction
MPIComm::get_instruction(const std::vector<std::array<int, 4>>& replicaMap) {
  // NOTE this varialbe is only significant on world root.
  std::vector<int> new_size_map;
  if (_world->rank() == _world_root) {
    new_size_map.resize(_cartesian->size());
    int num_avail_procs = 0;
    MPI_Group_size(_idle_procs, &num_avail_procs);
    std::fill(new_size_map.begin(), new_size_map.end(),
              1);  // 1 because at least there is a primary
    // cartesian's coord and dim arrays are of length of the actual dimension of
    // the simulation, so they can be short of 3.
    std::array<int, 3> cart_dim = {1, 1, 1};
    for (int i = 0; i < _cartesian->ndims(); ++i) {
      cart_dim[i] = _cartesian->dim(i);
    }

    for (const auto& item : replicaMap) {
      if (0 == num_avail_procs) break;
      // each item is of the format { coordx, coordy, coordz, num_replicas }
      int label = coord_to_ensemble_label(
          item[0], item[1], item[2], cart_dim[0], cart_dim[1], cart_dim[2]);
      if (label < _cartesian->size()) {  // this eliminates adding replicas to
                                         // non existing ensemble
        int num_replicas = std::min(item[3], num_avail_procs);
        new_size_map[label] += num_replicas;
        num_avail_procs -= num_replicas;
      }
    }
  }

  return decode_raw_instruction(get_raw_instruction(new_size_map));
}

std::vector<int>
MPIComm::get_raw_instruction(const std::vector<int>& size_map) {
  {
    // this function takes size_map and deploy processes accordingly.
    // NOTE that size_map is assumed to be valid, meaning that processes are
    // enough to carry out this size map.
    // see Logs for rules of the raw instruction.
    std::vector<int> result;

    if (_world->rank() == _world_root) {
      MPI_Group grp_world = _world->group();
      // define a function that puts the specified rank into _idle_procs
      auto f_put_to_idle =
          [& grp_idle = this->_idle_procs, &grp_world ](int rank) {
        // push to front
        MPI_Group grp_new_rank;
        MPI_Group_incl(grp_world, 1, &rank, &grp_new_rank);
        MPI_Group grp_idle_old = grp_idle;
        MPI_Group_union(grp_new_rank, grp_idle_old, &grp_idle);
        MPI_Group_free(&grp_new_rank);
        MPI_Group_free(&grp_idle_old);
      };

      // define a function that returns num_ranks amount of processes drawn from
      // _idle_procs
      auto f_draw_from_idle =
          [& grp_idle = this->_idle_procs, &grp_world ](int num_ranks) {
        std::vector<int> result(num_ranks);
        // pop from front
        // find the world ranks of first num_ranks processes in _idle_procs
        int* drawn_ranks = new int[num_ranks];
        for (int i = 0; i < num_ranks; ++i) drawn_ranks[i] = i;
        MPI_Group_translate_ranks(grp_idle, num_ranks, drawn_ranks, grp_world,
                                  result.data());
        // update _idle_procs
        MPI_Group grp_idle_old = grp_idle;
        MPI_Group_excl(grp_idle_old, num_ranks, drawn_ranks, &grp_idle);
        MPI_Group_free(&grp_idle_old);

        delete[] drawn_ranks;

        return result;
      };

      // define a function that does normal sends to desinations other than root
      // and copies to result if destination is root. Although MPI allows
      // sending and recving by same rank, this is necessary to prevent
      // deadlock.
      auto f_send_unless_root = [
        &result, &world = this->_world, root = this->_world_root
      ](int dest_rank, int tag, auto&& values, int n, MPI_Request& request) {
        if (dest_rank != root) {
          world->Isend(dest_rank, tag, std::forward<decltype(values)>(values),
                       n, request);
        } else {
          result.resize(n);
          std::copy_n(std::forward<decltype(values)>(values), n,
                      result.begin());
        }
      };

      // parse the size_map to get the following
      std::vector<int> shr_ens_labels;  // shrinking
      std::vector<int> exp_ens_labels;  // expanding
      std::vector<int> unv_ens_labels;  // unvarying
      const int num_ensembles = _label2ranks.size();
      for (int label = 0; label < num_ensembles; ++label) {
        int diff = size_map[label] - _label2ranks[label].size();
        if (diff > 0)
          exp_ens_labels.push_back(label);
        else if (diff < 0)
          shr_ens_labels.push_back(label);
        else
          unv_ens_labels.push_back(label);
      }

      // set up cache for instructions for leaving ranks in shrinking phase
      std::unordered_map<int, std::vector<int>> inst_cache;

      {  // do shrink first
        for (auto label : shr_ens_labels) {
          auto& members = _label2ranks[label];
          // extract processes from back
          int leave_size = members.size() - size_map[label];
          auto leave_begin = members.end() - leave_size;
          // prepare the instruction. 1. length of shrink information, including
          // the EnsAdjustCode 2. the EnsAdjustCode 3. the ranks that leave
          std::vector<int> instruction;

          {  // first deal with staying ranks, for which the instruciton is
             // complete and ready for communication.
            auto& inst_stay = instruction;
            inst_stay.push_back(1 +
                                leave_size);  // length of shrink information
            inst_stay.push_back(
                static_cast<int>(EnsAdjustCode::stay));  // EnsAdjustCode
            for (int j = 0; j < leave_size; ++j) {       // ranks that leave
              inst_stay.push_back(leave_begin[j]);
            }
            // for staying ranks, there is expanding action. So append the code
            // idle
            inst_stay.push_back(static_cast<int>(EnsAdjustCode::idle));

            auto requests = MPI_Helper::null_requests(
                std::distance(members.begin(), leave_begin));
            for (auto it = members.begin(); it < leave_begin; ++it) {
              f_send_unless_root(*it, *it, inst_stay.data(), inst_stay.size(),
                                 requests[std::distance(members.begin(), it)]);
            }
            _world->waitall(requests.size(), requests.data(),
                            MPI_STATUSES_IGNORE);
          }

          {  // next for leaving ranks, whose instruction must be cached
            // the instruction for leaving ranks differs by the EnsAdjustCode
            // and the expansion code should also be removed.
            auto& inst_leave = instruction;
            inst_leave[1] = static_cast<int>(EnsAdjustCode::move);
            inst_leave.pop_back();
            // when pushing to _idle_groups, start from the end
            for (auto it = members.end() - 1; it >= leave_begin; --it) {
              inst_cache.insert({*it, inst_leave});
              // put leaving ranks into _idle_procs
              f_put_to_idle(*it);
              // update a roster variable at this ensemble
              _rank2label[*it] = boost::none;
            }
            // update _label2ranks here in order not to interfere with above
            for (int i = 0; i < leave_size; ++i) members.pop_back();
          }
        }
      }

      {  // do expand next
        for (auto label : exp_ens_labels) {
          auto& members = _label2ranks[label];
          auto join_ranks = f_draw_from_idle(size_map[label] - members.size());

          std::vector<int> instruction;
          {  // first deal with staying ranks, for which the instuction is
             // complete and ready for communciation.
            auto& inst_stay = instruction;
            // for staying ranks, the leaving action is not needed.
            inst_stay.push_back(1);
            inst_stay.push_back(static_cast<int>(EnsAdjustCode::idle));
            // then put stay as expansion code
            inst_stay.push_back(static_cast<int>(EnsAdjustCode::stay));
            // put joining ranks into the instruction
            for (auto rank : join_ranks) inst_stay.push_back(rank);

            auto requests = MPI_Helper::null_requests(members.size());
            int req_idx = 0;
            for (auto rank : members)
              f_send_unless_root(rank, rank, inst_stay.data(), inst_stay.size(),
                                 requests[req_idx++]);
            _world->waitall(requests.size(), requests.data(),
                            MPI_STATUSES_IGNORE);
          }

          {  // next deal with joining ranks. Be careful if the joining rank
             // performed leaving action
            // first create an instruction that involves idle in the leaving
            // phase
            auto& inst_join = instruction;
            inst_join.resize(3);
            inst_join[0] = 1;
            inst_join[1] = static_cast<int>(EnsAdjustCode::idle);
            inst_join[2] = static_cast<int>(EnsAdjustCode::move);
            // NOTE the following information must be in the specified order.
            // Everything refers to the new ensemble
            inst_join.push_back(label);  // label
            inst_join.push_back(
                members[0]);  // root, which is the first element
            inst_join.push_back(members.size());  // number of staying members
            // list of staying members
            for (auto rank : members) inst_join.push_back(rank);
            // list of joining members
            for (auto rank : join_ranks) inst_join.push_back(rank);

            // communicate. For joining ranks cached with leaving instructions,
            // append the appropriate materials to the instruction
            auto requests = MPI_Helper::null_requests(join_ranks.size());
            int req_idx = 0;
            for (auto rank : join_ranks) {
              // if existing in cache
              if (inst_cache.find(rank) != inst_cache.end()) {
                auto& inst = inst_cache[rank];
                for (auto it = inst_join.begin() + 2; it != inst_join.end();
                     ++it)
                  inst.push_back(*it);

                f_send_unless_root(rank, rank, inst.data(), inst.size(),
                                   requests[req_idx++]);

              } else {
                f_send_unless_root(rank, rank, inst_join.data(),
                                   inst_join.size(), requests[req_idx++]);
              }

              // update _label2ranks at this ensemble
              members.push_back(rank);
              _rank2label[rank] = label;
            }
            _world->waitall(requests.size(), requests.data(),
                            MPI_STATUSES_IGNORE);
            // remove the current rank from cache. NOTE that we do it after
            // wailall rathe than in the loop in order not to alter the Isend
            // buffer
            for (auto rank : join_ranks) {
              // if existing in cache
              if (inst_cache.find(rank) != inst_cache.end())
                inst_cache.erase(rank);
            }
          }
        }
      }

      {  // do unvary next. No need to update _label2ranks and _rank2label
        std::vector<int> inst_unv(3);
        inst_unv[0] = 1;
        inst_unv[1] = static_cast<int>(EnsAdjustCode::idle);
        inst_unv[2] = static_cast<int>(EnsAdjustCode::idle);
        // count the number of such processes in order to initialize requests
        // array
        int num_unv_procs = 0;
        for (auto label : unv_ens_labels)
          num_unv_procs += _label2ranks[label].size();

        auto requests = MPI_Helper::null_requests(num_unv_procs);
        int req_idx = 0;
        for (auto label : unv_ens_labels) {
          for (auto rank : _label2ranks[label]) {
            f_send_unless_root(rank, rank, inst_unv.data(), inst_unv.size(),
                               requests[req_idx++]);
          }
        }
        _world->waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
      }

      {  // lastly, deal with processes in the _idle_procs. NOTE that some
         // processes may have cached instruction. NOTE _label2ranks and
         // _rank2label are already updated.
        std::vector<int> inst_idle(3);
        inst_idle[0] = 1;
        inst_idle[1] = static_cast<int>(EnsAdjustCode::idle);
        inst_idle[2] = static_cast<int>(EnsAdjustCode::idle);

        int num_idle_procs = 0;
        MPI_Group_size(_idle_procs, &num_idle_procs);

        auto requests = MPI_Helper::null_requests(num_idle_procs);
        for (int grp_rank = 0; grp_rank < num_idle_procs; ++grp_rank) {
          int world_rank = 0;
          MPI_Group_translate_ranks(_idle_procs, 1, &grp_rank, grp_world,
                                    &world_rank);
          // if existing in cache
          if (inst_cache.find(world_rank) != inst_cache.end()) {
            auto& inst = inst_cache[world_rank];
            inst.push_back(static_cast<int>(EnsAdjustCode::idle));
            f_send_unless_root(world_rank, world_rank, inst.data(), inst.size(),
                               requests[grp_rank]);
          } else {
            f_send_unless_root(world_rank, world_rank, inst_idle.data(),
                               inst_idle.size(), requests[grp_rank]);
          }
        }
        _world->waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
      }

      MPI_Group_free(&grp_world);
    }

    else {  // not world_root
      MPI_Status status = _world->probe(_world_root, _world->rank());
      int count = _world->get_count(result.data(), &status);
      result.resize(count);
      _world->recv(_world_root, _world->rank(), result.data(), result.size());
    }

    return result;
  }
}

EnsAdjustInstruction
MPIComm::decode_raw_instruction(const std::vector<int>& raw_instruction) const {
  // define some varialbes for both shrinking and expanding information
  // Note that info_begion excludes the EnsAdjustCode
  const auto shr_code = static_cast<EnsAdjustCode>(raw_instruction[1]);
  const auto* const shr_info_begin = raw_instruction.data() + 2;
  int shr_info_size =
      raw_instruction[0] - 1;  // -1 to exclude the EnsAdjustCode

  const auto exp_code =
      static_cast<EnsAdjustCode>(raw_instruction[raw_instruction[0] + 1]);
  const auto* const exp_info_begin =
      raw_instruction.data() + raw_instruction[0] + 2;
  int exp_info_size = raw_instruction.size() - raw_instruction[0] - 2;

  // sanity check
  if ((EnsAdjustCode::idle == shr_code) != (shr_info_size == 0))
    throw std::runtime_error("Inconsistent shrinking message!");
  if ((EnsAdjustCode::idle == exp_code) != (exp_info_size == 0))
    throw std::runtime_error("Inconsistent expanding message!");

  EnsAdjustInstruction result;

  // NOTE: when decoding, the old ensemble is still there, so the information
  // about it is still valid
  MPI_Group grp_world = _world->group();
  MPI_Group grp_ens = _ensemble->group();  // possibly empty for an idle process

  // define a helper function for inheriting old ensemble label and root
  auto f_ens_inherit = [&grp_world, &grp_ens, &ensemble = this->ensemble() ](
      EnsAdjustInstruction & inst) {
    inst._new_ens_label = ensemble.label();
    int root_ens_rank = ensemble.root_ensemble();
    int root_world_rank = 0;
    MPI_Group_translate_ranks(grp_ens, 1, &root_ens_rank, grp_world,
                              &root_world_rank);
    inst._new_ens_root = root_world_rank;
  };

  // decode info of shriking phase behavior
  result._is_shrink = (shr_info_size > 0);

  if (result._is_shrink) {
    if (EnsAdjustCode::stay == shr_code) {
      f_ens_inherit(result);
    }

    // except for setting ens label and ens root, stay and move are treated the
    // same way in the case of shrinking
    result._shrink_leave_ranks.resize(shr_info_size);
    for (int i = 0; i < shr_info_size; ++i) {
      result._shrink_leave_ranks[i] = shr_info_begin[i];
    }

    // use local ensemble to get the ranks of staying processes
    result._shrink_stay_ranks.resize(_ensemble->size() -
                                     result._shrink_leave_ranks.size());
    int idx = 0;
    for (int i = 0; i < _ensemble->size(); ++i) {
      int world_rank = 0;
      MPI_Group_translate_ranks(grp_ens, 1, &i, grp_world, &world_rank);
      // check if this process is one that leaves.
      if (std::find(result._shrink_leave_ranks.begin(),
                    result._shrink_leave_ranks.end(),
                    world_rank) == result._shrink_leave_ranks.end()) {
        // the process is staying
        result._shrink_stay_ranks[idx] = world_rank;
        ++idx;
      }
    }
  }

  // decode info of expanding phase behavior
  result._is_expand = (exp_info_size > 0);
  if (result._is_expand) {
    switch (exp_code) {
      case EnsAdjustCode::stay: {
        f_ens_inherit(result);

        result._expand_num_stays = _ensemble->size();
        // NOTE the order when initializing _expand_total_ranks
        result._expand_total_ranks.resize(_ensemble->size() + exp_info_size);
        for (int i = 0; i < _ensemble->size(); ++i) {
          int world_rank = 0;
          MPI_Group_translate_ranks(grp_ens, 1, &i, grp_world, &world_rank);
          result._expand_total_ranks[i] = world_rank;
        }
        for (int j = 0; j < exp_info_size; ++j) {
          result._expand_total_ranks[_ensemble->size() + j] = exp_info_begin[j];
        }
        break;
      }

      case EnsAdjustCode::move: {
        result._new_ens_label = exp_info_begin[0];
        result._new_ens_root = exp_info_begin[1];
        result._expand_num_stays = exp_info_begin[2];
        result._expand_total_ranks.resize(exp_info_size - 3);
        for (int i = 0; i < exp_info_size - 3; ++i) {
          result._expand_total_ranks[i] = exp_info_begin[3 + i];
        }
        break;
      }

      default:;
    }
  }

  // an ad hoc variable _is_retire
  result._is_retire =
      (EnsAdjustCode::move == shr_code && EnsAdjustCode::idle == exp_code);

  MPI_Group_free(&grp_world);
  if (grp_ens != MPI_GROUP_EMPTY) MPI_Group_free(&grp_ens);
  return result;
}
}  // namespace Aperture

#endif  // _MPI_COMM_IMPL_H_
