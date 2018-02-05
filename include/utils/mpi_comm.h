#ifndef _MPI_COMM_H_
#define _MPI_COMM_H_

#include "data/vec3.h"
#include "utils/mpi_helper.h"
#include <boost/optional.hpp>
#include <memory>
#include <mpi.h>
#include <string>
#include <vector>

#define NEIGHBOR_NULL -1

namespace Aperture {

class EnsAdjustInstruction;

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
  ///  Various recv methods, have to be used with send in a matching form
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
  // somehow, MPI_Get_count doesn't take the first parameter as const. It's
  // better to
  // use const MPIDatatypedStatus
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
  // recvcount refers to counts of recved data from one single process, rather
  // than the total
  // counts from all processes.
  template <typename T>
  void gather(const T* send_buf, int sendcount, T* recv_buf, int recvcount,
              int root) const;

  template <typename T>
  void all_gather(const T* send_buf, int sendcount, T* recv_buf,
                  int recvcount) const;

  // this version is mostly used by non-root processes becuase in this case
  // recv_buf and recvcount are not significant
  template <typename T>
  void gather(const T* send_buf, int sendcount, int root) const;

  // this version is called by root in an in-place manner
  template <typename T>
  void gather_inplace(T* recv_buf, int recvcount, int root) const;

  template <typename T>
  void gatherv(const T* send_buf, int sendcount, T* recv_buf,
               const int* recvcounts, const int* displs, int root) const;

  // this version is mostly used by non-root processes becuase in this case
  // recv_buf and recvcount are not significant
  template <typename T>
  void gatherv(const T* send_buf, int sendcount, int root) const;

  // this version is called by root in an in-place manner
  template <typename T>
  void gatherv_inplace(T* recv_buf, const int* recvcounts, const int* displs,
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
  int get_count(const T* value, const MPI_Status* status) const;

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
    // NOTE calling MPI_Comm_group on MPI_COMM_NULL is invalid, hence this
    // branch.
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
                   const std::vector<int>& ranks);

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
  inline MPI_Comm comm_row(int direction) const { return _rows[direction]; }

  inline int ndims() const { return _ndims; }
  inline int dim(int direction) const { return _dims[direction]; }
  inline int neighbor_right(int direction) const {
    return _neighbor_right[direction];
  }
  inline int neighbor_left(int direction) const {
    return _neighbor_left[direction];
  }
  inline int neighbor_corner(int id) const { return _neighbor_corner[id]; }
  inline int coord(int direction) const { return _coords[direction]; }
};  // ----- end of class mpi_cartesian -----

class MPICommEnsemble : public MPICommBase {
  int _root_ensemble = MPI_UNDEFINED;  // the primary in this ensemble
  int _label = MPI_UNDEFINED;          // unique label given to each ensemble

 public:
  MPICommEnsemble();
  virtual ~MPICommEnsemble() {}

  void create_ensemble(int label, int root_rank_world,
                       const std::vector<int>& members);
  void nullify();

  inline int root_ensemble() const { return _root_ensemble; }
  inline int label() const { return _label; }
};  // ----- end of class mpi_ensemble -----

class MPIComm {
  std::unique_ptr<MPICommWorld> _world;
  std::unique_ptr<MPICommCartesian> _cartesian;
  std::unique_ptr<MPICommEnsemble> _ensemble;

  const int _world_root = 0;

  MPI_Group _idle_procs = MPI_GROUP_EMPTY;
  std::vector<std::vector<int>> _label2ranks;  // index is ensemble label, value
                                               // is a list of world ranks
                                               // members belonging to that
                                               // ensemble, the first of which
                                               // is the root of the ensemble.
  std::vector<boost::optional<int>>
      _rank2label;  // the ith element is the ensemble label of world rank i

  std::vector<int> get_raw_instruction(const std::vector<int>& size_map);
  EnsAdjustInstruction decode_raw_instruction(
      const std::vector<int>& raw_instruction) const;

  inline int coord_to_ensemble_label(int coordx, int coordy, int coordz,
                                     int cart_dimx, int cart_dimy,
                                     int cart_dimz) {
    return coordx + coordy * cart_dimx + coordz * cart_dimx * cart_dimy;
  }

 public:
  MPIComm();
  MPIComm(int* argc, char*** argv);
  ~MPIComm();

  // Deploying processes for creating communicators. The functions return all
  // process ranks of the resulting communicator that includes the calling
  // process
  // NOTE that RVO is used in implementation
  std::vector<int> get_cartesian_members(int cart_size);
  // this function create a trivial ensemble that contains only the primary.
  // Meanwhile, it will let primaries communicate their ensemble labels (
  // generated by cartesian coords ) to _world_root so that the latter can
  // initialize _label2ranks and _rank2label.
  void init_trivial_ensemble();

  // get the ensemble adjustment instruction of the calling process
  // Ver 1. achieved through gathering number of particles onto world root
  EnsAdjustInstruction get_instruction(size_t num_ptc,
                                       size_t target_num_ptc_per_proc);
  // Ver 2. achieved through reading from a prescribed replicaMap.
  // NOTE: replicaMap only specifies the number of replicas
  EnsAdjustInstruction get_instruction(
      const std::vector<std::array<int, 4>>& replicaMap);

  inline const MPICommWorld& world() const { return *_world; }
  inline MPICommWorld& world() { return *_world; }
  inline const MPICommCartesian& cartesian() const { return *_cartesian; }
  inline MPICommCartesian& cartesian() { return *_cartesian; }
  inline const MPICommEnsemble& ensemble() const { return *_ensemble; }
  inline MPICommEnsemble& ensemble() { return *_ensemble; }

  inline const std::vector<std::vector<int>>& label2ranks() const {
    if (_world->rank() != _world_root)
      throw std::runtime_error("None-world-root tries to access label2ranks!");
    return _label2ranks;
  }

  inline const std::vector<boost::optional<int>>& rank2label() const {
    if (_world->rank() != _world_root)
      throw std::runtime_error("None-world-root tries to access rank2label!");
    return _rank2label;
  }

  // get the world ranks of the members in the given group
  std::vector<int> group2ranks(MPI_Group group) const;

  std::vector<int> idle_ranks() const;

  // return the group of active processes, namely primary or replica
  std::vector<int> active_procs() const;

  inline int world_root() const { return _world_root; }
  inline bool is_world_root() const { return _world->rank() == _world_root; }

  friend class Test_mpi;
};

class EnsAdjustInstruction {
 private:
  bool _is_shrink = false;  // whether participate in shrinking communication
  bool _is_expand = false;  // whether participate in expanding communication
  bool _is_retire = false;  // whether leaves from a shrinking ensemble and
                            // becomes idle. This is an ad hoc fix in order to
                            // correctly implement NewListOfEnsMembersIfAdjusted
                            // and to signal relevant processes to nullify their
                            // ensemble communicator.

  boost::optional<int> _new_ens_label;
  boost::optional<int> _new_ens_root;
  std::vector<int>
      _shrink_stay_ranks;  // all ranks that stay in a shrinking ensemble
  std::vector<int>
      _shrink_leave_ranks;  // all ranks that leave a shrinking ensemble
  std::vector<int> _expand_total_ranks;  // all ranks in an expanded ensemble in
                                         // the order of those that stay
                                         // followed by those that join
  int _expand_num_stays =
      0;  // number of ranks that stay in the expanded ensemble

  EnsAdjustInstruction() {}

 public:
  inline bool IsShrink() const { return _is_shrink; }
  inline bool IsExpand() const { return _is_expand; }
  inline bool IsAdjusted() const { return IsShrink() || IsExpand(); }
  inline bool IsRetire() const { return _is_retire; }

  // the following IfAdjusted-postfixed functions are NOT meant to be used by
  // idle processes and processes in unvarying ensembles.
  inline const boost::optional<int>& NewEnsLabelIfAdjusted() const {
    return _new_ens_label;
  }
  inline const boost::optional<int>& NewEnsRootIfAdjusted() const {
    return _new_ens_root;
  }
  const std::vector<int>& NewListOfEnsMembersIfAdjusted() const {
    // If the process does expanding action, always use _expand_total_ranks.
    // Else we are left with the following cases: stay in shrink, retire from
    // shrink, stay in unvary, remain idle. All except retiring can use
    // _shrink_stay_ranks as the return value. The retiring case should return
    // {}, which happens to be its _expand_total_ranks.
    return (IsExpand() || IsRetire()) ? _expand_total_ranks
                                      : _shrink_stay_ranks;
  }

  // parameters for performing shrinking ensemble actions
  const int* Shrink_Stay_Begin() const { return _shrink_stay_ranks.data(); }
  int Shrink_Stay_Num() const { return _shrink_stay_ranks.size(); }
  const int* Shrink_Stay_End() const {
    return Shrink_Stay_Begin() + Shrink_Stay_Num();
  }

  const int* Shrink_Leave_Begin() const { return _shrink_leave_ranks.data(); }
  int Shrink_Leave_Num() const { return _shrink_leave_ranks.size(); }
  const int* Shrink_Leave_End() const {
    return Shrink_Leave_Begin() + Shrink_Leave_Num();
  }

  // parameters for performing expanding ensemble actions
  const int* Expand_Stay_Begin() const { return _expand_total_ranks.data(); }
  int Expand_Stay_Num() const { return _expand_num_stays; }
  const int* Expand_Stay_End() const {
    return Expand_Stay_Begin() + Expand_Stay_Num();
  }

  const int* Expand_Join_Begin() const {
    return _expand_total_ranks.data() + _expand_num_stays;
  }
  int Expand_Join_Num() const {
    return _expand_total_ranks.size() - _expand_num_stays;
  }
  const int* Expand_Join_End() const {
    return Expand_Join_Begin() + Expand_Join_Num();
  }

  friend class MPIComm;
  // friend class TestParticleCommunicator;
};
}

// #include "utils/mpi_comm_impl.hpp"

#endif  // ----- #ifndef _MPI_COMM_H_  -----
