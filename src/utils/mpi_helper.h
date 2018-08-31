#ifndef _MPI_HELPER_H_
#define _MPI_HELPER_H_

#include <vector>
#include <mpi.h>

namespace Aperture {

class MPICommBase;

extern MPI_Datatype MPI_VEC3_FLOAT;
extern MPI_Datatype MPI_VEC3_DOUBLE;
extern MPI_Datatype MPI_VEC3_INT;
extern MPI_Datatype MPI_VEC3_CHAR;
extern MPI_Datatype MPI_PARTICLES;
extern MPI_Datatype MPI_PHOTONS;

namespace MPI_Helper {

template <typename T>
MPI_Datatype get_mpi_datatype(const T& x);

void register_types();

void free_types();

void handle_mpi_error(int error_code, int rank);

void handle_mpi_error(int error_code, const MPICommBase& mpi);

std::vector<MPI_Request> null_requests(int size);

}
}

#endif  // _MPI_HELPER_H_
