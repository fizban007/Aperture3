#include "utils/mpi_helper.h"
#include <stddef.h>
#include <boost/fusion/include/size.hpp>
#include "data/particle_data.h"
#include "data/vec3.h"
#include "utils/mpi_comm.h"

#define BUFSIZE 1024

namespace Aperture {

MPI_Datatype MPI_VEC3_FLOAT;
MPI_Datatype MPI_VEC3_DOUBLE;
MPI_Datatype MPI_VEC3_INT;
MPI_Datatype MPI_VEC3_CHAR;
MPI_Datatype MPI_VEC4_FLOAT;
MPI_Datatype MPI_VEC4_DOUBLE;
MPI_Datatype MPI_VEC4_INT;
MPI_Datatype MPI_VEC4_CHAR;
MPI_Datatype MPI_PARTICLES;
MPI_Datatype MPI_PHOTONS;

namespace MPI_Helper {

////////////////////////////////////////////////////////////////////////////////
///  Specialize the MPI built-in data types
////////////////////////////////////////////////////////////////////////////////
template <>
MPI_Datatype get_mpi_datatype(const char& x) {
  return MPI_CHAR;
}

template <>
MPI_Datatype get_mpi_datatype(const short& x) {
  return MPI_SHORT;
}

template <>
MPI_Datatype get_mpi_datatype(const int& x) {
  return MPI_INT;
}

template <>
MPI_Datatype get_mpi_datatype(const uint32_t& x) {
  return MPI_UINT32_T;
}

template <>
MPI_Datatype get_mpi_datatype(const uint16_t& x) {
  return MPI_UINT16_T;
}

template <>
MPI_Datatype get_mpi_datatype(const bool& x) {
  return MPI::BOOL;
}

template <>
MPI_Datatype get_mpi_datatype(const long& x) {
  return MPI_LONG;
}

template <>
MPI_Datatype get_mpi_datatype(const unsigned char& x) {
  return MPI_UNSIGNED_CHAR;
}

// template<>
// MPI_Datatype get_mpi_datatype(const unsigned short& x) { return
// MPI_UNSIGNED_SHORT; }

// template<>
// MPI_Datatype get_mpi_datatype(const unsigned int& x) { return MPI_UNSIGNED; }

template <>
MPI_Datatype get_mpi_datatype(const unsigned long& x) {
  return MPI_UNSIGNED_LONG;
}

template <>
MPI_Datatype get_mpi_datatype(const float& x) {
  return MPI_FLOAT;
}

template <>
MPI_Datatype get_mpi_datatype(const double& x) {
  return MPI_DOUBLE;
}

template <>
MPI_Datatype get_mpi_datatype(const long double& x) {
  return MPI_LONG_DOUBLE;
}

template <>
MPI_Datatype get_mpi_datatype(const Vec3<float>& x) {
  return MPI_VEC3_FLOAT;
}

template <>
MPI_Datatype get_mpi_datatype(const Vec3<double>& x) {
  return MPI_VEC3_DOUBLE;
}

template <>
MPI_Datatype get_mpi_datatype(const Vec3<int>& x) {
  return MPI_VEC3_INT;
}

template <>
MPI_Datatype get_mpi_datatype(const Vec3<char>& x) {
  return MPI_VEC3_CHAR;
}

template <>
MPI_Datatype get_mpi_datatype(const Vec4<float>& x) {
  return MPI_VEC4_FLOAT;
}

template <>
MPI_Datatype get_mpi_datatype(const Vec4<double>& x) {
  return MPI_VEC4_DOUBLE;
}

template <>
MPI_Datatype get_mpi_datatype(const Vec4<int>& x) {
  return MPI_VEC4_INT;
}

template <>
MPI_Datatype get_mpi_datatype(const Vec4<char>& x) {
  return MPI_VEC4_CHAR;
}

template <>
MPI_Datatype get_mpi_datatype(const single_particle_t& x) {
  return MPI_PARTICLES;
}

template <>
MPI_Datatype get_mpi_datatype(const single_photon_t& x) {
  return MPI_PHOTONS;
}

template <typename Type>
void register_vec3_type(const Type& t, MPI_Datatype* type) {
  const int n_entries = 3;
  int blocklengths[n_entries] = {1, 1, 1};
  MPI_Datatype mt = get_mpi_datatype(t);
  MPI_Datatype types[n_entries] = {mt, mt, mt};
  MPI_Aint offsets[n_entries] = {offsetof(Vec3<Type>, x),
                                 offsetof(Vec3<Type>, y),
                                 offsetof(Vec3<Type>, z)};
  // std::cout << offsets[0] << " " << offsets[1] << " " << offsets[2] <<
  // std::endl;

  MPI_Type_create_struct(n_entries, blocklengths, offsets, types, type);
  MPI_Type_commit(type);
  // _data_types.push_back(target_type);
}

template <typename Type>
void register_vec4_type(const Type& t, MPI_Datatype* type) {
  const int n_entries = 4;
  int blocklengths[n_entries] = {1, 1, 1, 1};
  MPI_Datatype mt = get_mpi_datatype(t);
  MPI_Datatype types[n_entries] = {mt, mt, mt, mt};
  MPI_Aint offsets[n_entries] = {
      offsetof(Vec4<Type>, x), offsetof(Vec4<Type>, y), offsetof(Vec4<Type>, z),
      offsetof(Vec4<Type>, w)};
  // std::cout << offsets[0] << " " << offsets[1] << " " << offsets[2] <<
  // std::endl;

  MPI_Type_create_struct(n_entries, blocklengths, offsets, types, type);
  MPI_Type_commit(type);
  // _data_types.push_back(target_type);
}

template <typename ParticleType>
void register_particle_type(const ParticleType& p_def, MPI_Datatype* type) {
  constexpr int n_entries =
      boost::fusion::result_of::size<ParticleType>::type::value;
  int blocklengths[n_entries];
  MPI_Datatype types[n_entries];
  MPI_Aint offsets[n_entries];

  int n = 0;
  int offset = 0;
  boost::fusion::for_each(
      p_def, [&n, &offset, &blocklengths, &types, &offsets](auto& x) {
        blocklengths[n] = 1;
        types[n] = get_mpi_datatype(
            typename std::remove_reference<decltype(x)>::type());
        offsets[n] = offset;
        n += 1;
        offset += sizeof(typename std::remove_reference<decltype(x)>::type);
      });

  MPI_Type_create_struct(n_entries, blocklengths, offsets, types, type);
  MPI_Type_commit(type);
}

void register_types() {
  register_vec3_type(float(), &MPI_VEC3_FLOAT);
  register_vec3_type(double(), &MPI_VEC3_DOUBLE);
  register_vec3_type(int(), &MPI_VEC3_INT);
  register_vec3_type(char(), &MPI_VEC3_CHAR);
  register_vec4_type(float(), &MPI_VEC4_FLOAT);
  register_vec4_type(double(), &MPI_VEC4_DOUBLE);
  register_vec4_type(int(), &MPI_VEC4_INT);
  register_vec4_type(char(), &MPI_VEC4_CHAR);
  register_particle_type(single_particle_t(), &MPI_PARTICLES);
  register_particle_type(single_photon_t(), &MPI_PHOTONS);
}

void free_types() {
  MPI_Type_free(&MPI_VEC3_FLOAT);
  MPI_Type_free(&MPI_VEC3_DOUBLE);
  MPI_Type_free(&MPI_VEC3_INT);
  MPI_Type_free(&MPI_VEC3_CHAR);
  MPI_Type_free(&MPI_VEC4_FLOAT);
  MPI_Type_free(&MPI_VEC4_DOUBLE);
  MPI_Type_free(&MPI_VEC4_INT);
  MPI_Type_free(&MPI_VEC4_CHAR);
  MPI_Type_free(&MPI_PARTICLES);
  MPI_Type_free(&MPI_PHOTONS);
}

void handle_mpi_error(int error_code, int rank) {
  if (error_code != MPI_SUCCESS) {
    char error_string[BUFSIZE];
    int length_of_error_string;

    MPI_Error_string(error_code, error_string, &length_of_error_string);
    fprintf(stderr, "%3d: %s\n", rank, error_string);
  }
}

void handle_mpi_error(int error_code, const MPICommBase& comm) {
  if (error_code != MPI_SUCCESS) {
    char error_string[BUFSIZE];
    int length_of_error_string;

    MPI_Error_string(error_code, error_string, &length_of_error_string);
    fprintf(stderr, "%s rank %3d: %s\n", comm.name().c_str(), comm.rank(), error_string);
  }
}

std::vector<MPI_Request> null_requests(int size) {
  std::vector<MPI_Request> requests( size );
  std::fill(requests.begin(), requests.end(), MPI_REQUEST_NULL );
  return requests;
}


}
}
