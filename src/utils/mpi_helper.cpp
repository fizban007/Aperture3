#include "utils/mpi_helper.h"
#include <stddef.h>
// #include <boost/fusion/include/size.hpp>
#include "core/vec3.h"
#include "data/particle_data.h"
#include "visit_struct/visit_struct.hpp"
// #include "utils/mpi_comm.h"

#define BUFSIZE 1024

namespace Aperture {

MPI_Datatype MPI_VEC3_FLOAT;
MPI_Datatype MPI_VEC3_DOUBLE;
MPI_Datatype MPI_VEC3_INT;
MPI_Datatype MPI_VEC3_CHAR;
MPI_Datatype MPI_PARTICLE;
MPI_Datatype MPI_PHOTON;

namespace MPI_Helper {

struct def_mpi_struct {
  int n_, offset_;
  int* blocklengths_;
  MPI_Datatype* types_;
  MPI_Aint* offsets_;

  def_mpi_struct(int n, int offset, int blocklengths[],
                 MPI_Datatype types[], MPI_Aint offsets[])
      : n_(n),
        offset_(offset),
        blocklengths_(&blocklengths[0]),
        types_(&types[0]),
        offsets_(&offsets[0]) {}

  template <typename T>
  void operator()(const char* name, T& x) {
    blocklengths_[n_] = 1;
    types_[n_] = get_mpi_datatype(
        typename std::remove_reference<decltype(x)>::type());
    offsets_[n_] = offset_;
    n_ += 1;
    offset_ +=
        sizeof(typename std::remove_reference<decltype(x)>::type);
  }
};
////////////////////////////////////////////////////////////////////////////////
///  Specialize the MPI built-in data types
////////////////////////////////////////////////////////////////////////////////
template <>
MPI_Datatype
get_mpi_datatype(const char& x) {
  return MPI_CHAR;
}

template <>
MPI_Datatype
get_mpi_datatype(const short& x) {
  return MPI_SHORT;
}

template <>
MPI_Datatype
get_mpi_datatype(const int& x) {
  return MPI_INT;
}

template <>
MPI_Datatype
get_mpi_datatype(const uint32_t& x) {
  return MPI_UINT32_T;
}

template <>
MPI_Datatype
get_mpi_datatype(const uint16_t& x) {
  return MPI_UINT16_T;
}

template <>
MPI_Datatype
get_mpi_datatype(const bool& x) {
  return MPI_C_BOOL;
}

template <>
MPI_Datatype
get_mpi_datatype(const long& x) {
  return MPI_LONG;
}

template <>
MPI_Datatype
get_mpi_datatype(const unsigned char& x) {
  return MPI_UNSIGNED_CHAR;
}

// template<>
// MPI_Datatype get_mpi_datatype(const unsigned short& x) { return
// MPI_UNSIGNED_SHORT; }

// template<>
// MPI_Datatype get_mpi_datatype(const unsigned int& x) { return
// MPI_UNSIGNED; }

template <>
MPI_Datatype
get_mpi_datatype(const unsigned long& x) {
  return MPI_UNSIGNED_LONG;
}

template <>
MPI_Datatype
get_mpi_datatype(const float& x) {
  return MPI_FLOAT;
}

template <>
MPI_Datatype
get_mpi_datatype(const double& x) {
  return MPI_DOUBLE;
}

template <>
MPI_Datatype
get_mpi_datatype(const long double& x) {
  return MPI_LONG_DOUBLE;
}

template <>
MPI_Datatype
get_mpi_datatype(const Vec3<float>& x) {
  return MPI_VEC3_FLOAT;
}

template <>
MPI_Datatype
get_mpi_datatype(const Vec3<double>& x) {
  return MPI_VEC3_DOUBLE;
}

template <>
MPI_Datatype
get_mpi_datatype(const Vec3<int>& x) {
  return MPI_VEC3_INT;
}

template <>
MPI_Datatype
get_mpi_datatype(const Vec3<char>& x) {
  return MPI_VEC3_CHAR;
}

template <>
MPI_Datatype
get_mpi_datatype(const single_particle_t& x) {
  return MPI_PARTICLE;
}

template <>
MPI_Datatype
get_mpi_datatype(const single_photon_t& x) {
  return MPI_PHOTON;
}

template <typename Type>
void
register_vec3_type(const Type& t, MPI_Datatype* type) {
  const int n_entries = 3;
  int blocklengths[n_entries] = {1, 1, 1};
  MPI_Datatype mt = get_mpi_datatype(t);
  MPI_Datatype types[n_entries] = {mt, mt, mt};
  MPI_Aint offsets[n_entries] = {offsetof(Vec3<Type>, x),
                                 offsetof(Vec3<Type>, y),
                                 offsetof(Vec3<Type>, z)};
  // std::cout << offsets[0] << " " << offsets[1] << " " << offsets[2]
  // << std::endl;

  MPI_Type_create_struct(n_entries, blocklengths, offsets, types, type);
  MPI_Type_commit(type);
  // _data_types.push_back(target_type);
}

template <typename ParticleType>
void
register_particle_type(const ParticleType& p_def, MPI_Datatype* type) {
  constexpr int n_entries = visit_struct::field_count<ParticleType>();
  // boost::fusion::result_of::size<ParticleType>::type::value;
  int blocklengths[n_entries];
  MPI_Datatype types[n_entries];
  MPI_Aint offsets[n_entries];

  int n = 0;
  int offset = 0;
  ParticleType p;
  visit_struct::for_each(
      p, def_mpi_struct{n, offset, blocklengths, types, offsets});

  MPI_Type_create_struct(n_entries, blocklengths, offsets, types, type);
  MPI_Type_commit(type);
}

void
register_types() {
  register_vec3_type(float(), &MPI_VEC3_FLOAT);
  register_vec3_type(double(), &MPI_VEC3_DOUBLE);
  register_vec3_type(int(), &MPI_VEC3_INT);
  register_vec3_type(char(), &MPI_VEC3_CHAR);
  register_particle_type(single_particle_t(), &MPI_PARTICLE);
  register_particle_type(single_photon_t(), &MPI_PHOTON);
}

void
free_types() {
  MPI_Type_free(&MPI_VEC3_FLOAT);
  MPI_Type_free(&MPI_VEC3_DOUBLE);
  MPI_Type_free(&MPI_VEC3_INT);
  MPI_Type_free(&MPI_VEC3_CHAR);
  MPI_Type_free(&MPI_PARTICLE);
  MPI_Type_free(&MPI_PHOTON);
}

void
handle_mpi_error(int error_code, int rank) {
  if (error_code != MPI_SUCCESS) {
    char error_string[BUFSIZE];
    int length_of_error_string;

    MPI_Error_string(error_code, error_string, &length_of_error_string);
    fprintf(stderr, "%3d: %s\n", rank, error_string);
  }
}

// void handle_mpi_error(int error_code, const MPICommBase& comm) {
//   if (error_code != MPI_SUCCESS) {
//     char error_string[BUFSIZE];
//     int length_of_error_string;

//     MPI_Error_string(error_code, error_string,
//     &length_of_error_string); fprintf(stderr, "%s rank %3d: %s\n",
//     comm.name().c_str(), comm.rank(), error_string);
//   }
// }

std::vector<MPI_Request>
null_requests(int size) {
  std::vector<MPI_Request> requests(size);
  std::fill(requests.begin(), requests.end(), MPI_REQUEST_NULL);
  return requests;
}

}  // namespace MPI_Helper
}  // namespace Aperture
