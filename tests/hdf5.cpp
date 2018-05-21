#include "highfive/H5File.hpp"
#include "data/multi_array.h"
#include <vector>
#include <stdexcept>
#include <mpi.h>

using namespace Aperture;
namespace HF = HighFive;

int main(int argc, char *argv[])
{
  int mpi_rank, mpi_size;

  // initialize MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  MultiArray<float> arr(10);
  arr.assign((float)mpi_rank);
  try {
    HF::File file("test.h5", HF::File::ReadWrite | HF::File::Create | HF::File::Truncate,
                  HF::MPIOFileDriver(MPI_COMM_WORLD, MPI_INFO_NULL));

    std::vector<size_t> dims(1);
    dims[0] = 10 * mpi_size;

    // Create the dataset
    HF::DataSet dataset = file.createDataSet<float>("data", HF::DataSpace(dims));
    // std::vector<std::vector<float*>> ptrs(arr.height(), arr.depth());
    // for (int k = 0; k < arr.depth(); k++) {
    //   for (int j = 0; j < arr.height(); j++) {
    //     ptrs[j][k] = &arr(0, j, k);
    //   }
    // }
    dataset.select({10u * mpi_rank, 0, 0}, {(size_t)arr.width(), 1, 1}).write(arr.data());

  } catch (HF::Exception& err) {
    // catch and print any HDF5 error
    std::cerr << err.what() << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  MPI_Finalize();
  return 0;
}
