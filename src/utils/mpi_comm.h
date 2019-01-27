#ifndef _MPI_COMM_H_
#define _MPI_COMM_H_

#include "core/vec3.h"
#include "utils/mpi_helper.h"
#include <boost/optional.hpp>
#include <memory>
#include <mpi.h>
#include <string>
#include <vector>

#define NEIGHBOR_NULL -1

namespace Aperture {

class MPICommBase {
 protected:
  MPI_Comm _comm = MPI_COMM_NULL;  //< Underlying MPI communicator
  int _rank = MPI_UNDEFINED;       //< Rank of this process
  int _size = 0;      //< Total number of processes in this communicator
  std::string _name;  //< Name of this Comm

 public:
  MPICommBase();
  MPICommBase(MPI_Comm _comm);
  // mpi_base(int* argc, char*** argv);
  virtual ~MPICommBase();

  ////////////////////////////////////////////////////////////////////////////////
  ///  These functions are for debugging purposes
  ////////////////////////////////////////////////////////////////////////////////
  void print_rank() const;

  void barrier() const;

  ////////////////////////////////////////////////////////////////////////////////
  ///  Various send methods
  ////////////////////////////////////////////////////////////////////////////////
  template <typename T>
  void send(int dest_rank, int tag, const T* values, int n) const;

  template <typename T>
  void send(int dest_rank, int tag, const T& value) const;

  // void send(int dest_rank, int tag) const;

  template <typename T>
  void Isend(int dest_rank, int tag, const T* values, int n,
             MPI_Request& request) const;

  ////////////////////////////////////////////////////////////////////////////////
  ///  Various recv methods, have to be used with send in a matching
  ///  form
  ////////////////////////////////////////////////////////////////////////////////
  template <typename T>
  void recv(int source_rank, int tag, T* values, int n,
            MPI_Status& status) const;

  template <typename T>
  void recv(int source_rank, int tag, T* values, int n) const;

  template <typename T>
  void recv(int source_rank, int tag, T& value) const;

  // template <typename T>
  // void recv(int source_rank, int tag);

  template <typename T>
  void Irecv(int source_rank, int tag, T* values, int n,
             MPI_Request& request) const;

  // returns the count of elements received.
  // somehow, MPI_Get_count doesn't take the first parameter as const.
  // It's better to use const MPIDatatypedStatus
  void get_recv_count(MPI_Status& status, MPI_Datatype datatype,
                      int& count) const;

  ////////////////////////////////////////////////////////////////////////////////
  ///  Various blocking send-recv methods
  ////////////////////////////////////////////////////////////////////////////////
  template <typename T>
  void send_recv(int source_rank, const T* src_values, int dest_rank,
                 T* dest_values, int tag, int n) const;

  template <typename T>
  void send_recv(int source_rank, const T& src_value, int dest_rank,
                 T& dest_value, int tag) const;

  template <typename T>
  void send_recv(int source_rank, int dest_rank, int tag) const;

  ////////////////////////////////////////////////////////////////////////////////
  ///  Gather methods
  ////////////////////////////////////////////////////////////////////////////////
  // recvcount refers to counts of recved data from one single process,
  // rather than the total counts from all processes.
  template <typename T>
  void gather(const T* send_buf, int sendcount, T* recv_buf,
              int recvcount, int root) const;

  template <typename T>
  void all_gather(const T* send_buf, int sendcount, T* recv_buf,
                  int recvcount) const;

  // this version is mostly used by non-root processes becuase in this
  // case recv_buf and recvcount are not significant
  template <typename T>
  void gather(const T* send_buf, int sendcount, int root) const;

  // this version is called by root in an in-place manner
  template <typename T>
  void gather_inplace(T* recv_buf, int recvcount, int root) const;

  template <typename T>
  void gatherv(const T* send_buf, int sendcount, T* recv_buf,
               int* recvcounts, int* displs, int root) const;

  // this version is mostly used by non-root processes becuase in this
  // case recv_buf and recvcount are not significant
  template <typename T>
  void gatherv(const T* send_buf, int sendcount, int root) const;

  // this version is called by root in an in-place manner
  template <typename T>
  void gatherv_inplace(T* recv_buf, int* recvcounts, int* displs,
                       int root) const;

  ////////////////////////////////////////////////////////////////////////////////
  ///  Wait methods, used to block Isend and Irecv
  ////////////////////////////////////////////////////////////////////////////////
  void waitall(int length_of_array, MPI_Request* array_of_requests,
               MPI_Status* array_of_statuses) const;

  ////////////////////////////////////////////////////////////////////////////////
  ///  Probe
  ////////////////////////////////////////////////////////////////////////////////
  MPI_Status probe(int source, int tag) const;

  // returns the count of elements received.
  template <typename T>
  int get_count(const T* value, MPI_Status* status) const;

  ////////////////////////////////////////////////////////////////////////////////
  ///  Accessors
  ////////////////////////////////////////////////////////////////////////////////
  // inline MPI_Comm comm_world() const { return MPI_COMM_WORLD; }
  inline MPI_Comm comm() const { return _comm; }
  //    inline MPI_Comm comm_col() const { return _col; }

  inline int rank() const { return _rank; }
  inline int size() const { return _size; }
  inline bool is_null() const { return _comm == MPI_COMM_NULL; }
  inline std::string name() const { return _name; }
  inline MPI_Group group() const {
    // NOTE to use MPI_GROUP_EMPTY rather than MPI_GROUP_NULL
    MPI_Group grp = MPI_GROUP_EMPTY;
    // NOTE calling MPI_Comm_group on MPI_COMM_NULL is invalid, hence
    // this branch.
    if (!is_null()) MPI_Comm_group(_comm, &grp);
    return grp;
  }
};  // ----- end of class mpi_base -----

class MPICommWorld : public MPICommBase {
 public:
  MPICommWorld();
  virtual ~MPICommWorld() {}
};  // ----- end of class mpi_world -----

class MPICommCartesian : public MPICommBase {
 private:
  int _ndims = 0;
  int* _dims = nullptr;
  bool* _periodic = nullptr;
  int* _coords = nullptr;

  int* _neighbor_right = nullptr;
  int* _neighbor_left = nullptr;
  int* _neighbor_corner = nullptr;
  MPI_Comm* _rows = nullptr;
  //    MPI_Comm _col;

 public:
  MPICommCartesian();
  virtual ~MPICommCartesian();

  void create_dims(int num_nodes, int ndims, int* dims);
  void create_cart(int ndims, int dims[], bool periodic[],
                   std::vector<int>& ranks);

  void printCoord() const;

  ////////////////////////////////////////////////////////////////////////////////
  ///  Scan methods
  ////////////////////////////////////////////////////////////////////////////////
  template <typename T>
  void scan(const T* send_buf, T* result_buf, int num, int scan_dir,
            bool exclusive = true) const;

  ////////////////////////////////////////////////////////////////////////////////
  ///  Accessors
  ////////////////////////////////////////////////////////////////////////////////
  inline MPI_Comm comm_row(int direction) const {
    return _rows[direction];
  }

  inline int ndims() const { return _ndims; }
  inline int dim(int direction) const { return _dims[direction]; }
  inline int neighbor_right(int direction) const {
    return _neighbor_right[direction];
  }
  inline int neighbor_left(int direction) const {
    return _neighbor_left[direction];
  }
  inline int neighbor_corner(int id) const {
    return _neighbor_corner[id];
  }
  inline int coord(int direction) const { return _coords[direction]; }
};  // ----- end of class mpi_cartesian -----

class MPIComm {
  std::unique_ptr<MPICommWorld> _world;
  std::unique_ptr<MPICommCartesian> _cartesian;

  const int _world_root = 0;

 public:
  MPIComm();
  MPIComm(int* argc, char*** argv);
  ~MPIComm();

  // Deploying processes for creating communicators. The functions
  // return all process ranks of the resulting communicator that
  // includes the calling process NOTE that RVO is used in
  // implementation
  std::vector<int> get_cartesian_members(int cart_size);

  inline const MPICommWorld& world() const { return *_world; }
  inline MPICommWorld& world() { return *_world; }
  inline const MPICommCartesian& cartesian() const {
    return *_cartesian;
  }
  inline MPICommCartesian& cartesian() { return *_cartesian; }

  inline int world_root() const { return _world_root; }
  inline bool is_world_root() const {
    return _world->rank() == _world_root;
  }
};

}  // namespace Aperture

// #include "utils/mpi_comm_impl.hpp"

#endif  // ----- #ifndef _MPI_COMM_H_  -----
