#include "utils/hdf_wrapper.h"
#include "sim_environment.h"
#include <iostream>

using namespace Aperture;

int main(int argc, char *argv[]) {
  sim_environment env(&argc, &argv);

  H5File file = hdf_create("hdftest.h5", H5CreateMode::trunc_parallel);
  uint64_t n = env.domain_info().rank;
  uint64_t size = env.domain_info().size;

  file.write_parallel(&n, 1, size, n, 1, 0, "num");

  uint64_t n_out;
  file.read_subset(&n_out, 1, "num", n, 1, 0);
  std::cout << n_out << std::endl;
  return 0;
}
