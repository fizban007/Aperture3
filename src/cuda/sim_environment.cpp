#include "sim_environment_impl.hpp"
#include "cuda/constant_mem_func.h"
#include <cuda_runtime.h>

namespace Aperture {

void
sim_environment::setup_env_extra() {
  // Poll the system to detect how many GPUs are on the node
  int n_devices;
  cudaGetDeviceCount(&n_devices);
  if (n_devices <= 0) {
    Logger::err("No usable Cuda device found!!");
    exit(1);
  }
  int rank = m_comm->world().rank();
  m_dev_id = rank % n_devices;

  // m_dev_map.resize(n_devices);
  Logger::print_info("Found {} Cuda devices, using dev {}", n_devices,
                     m_dev_id);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, m_dev_id);
  Logger::print_info("    Device Number: {}", m_dev_id);
  Logger::print_info("    Device Name: {}", prop.name);
  Logger::print_info("    Device Total Memory: {}MiB",
                     prop.totalGlobalMem / (1024 * 1024));

  init_dev_params(m_params);
  init_dev_mesh(m_grid->mesh());

  init_dev_charges(m_charges.data());
  init_dev_masses(m_masses.data());
}


}
