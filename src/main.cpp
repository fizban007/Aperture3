// #include "additional_diagnostics.h"
// #include "algorithms/field_solver_log_sph.h"
// #include "cuda/constant_mem_func.h"
// #include "cuda/cudarng.h"
// #include "ptc_updater_logsph.h"
// #include "radiation/curvature_instant.h"
// #include "radiation/radiation_transfer.h"
// #include "radiation/rt_pulsar.h"
// #include "cu_sim_data.h"
#include "sim_environment.h"
#include "sim_data.h"
#include "utils/logger.h"
#include "utils/timer.h"
#include "utils/util_functions.h"
#include <random>

using namespace Aperture;

int
main(int argc, char* argv[]) {
  uint32_t step = 0;
  // Construct the simulation environment
  sim_environment env(&argc, &argv);

  // Allocate simulation data
  sim_data data(env);

  // Initialize the field solver
  return 0;
}
