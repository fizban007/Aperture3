// #include "additional_diagnostics.h"
// #include "algorithms/field_solver_log_sph.h"
// #include "cuda/constant_mem_func.h"
// #include "cuda/cudarng.h"
// #include "ptc_updater_logsph.h"
// #include "radiation/curvature_instant.h"
// #include "radiation/radiation_transfer.h"
// #include "radiation/rt_pulsar.h"
// #include "cu_sim_data.h"
// #include "cuda/core/additional_diagnostics.h"
// #include "cuda/core/cu_sim_data.h"
// #include "cuda/core/field_solver_log_sph.h"
// #include "cuda/core/ptc_updater_logsph.h"
// #include "cuda/core/cu_sim_environment.h"
// #include "cuda/radiation/rt_pulsar.h"
// #include "cuda/utils/cu_data_exporter.h"
// #include "utils/data_exporter.h"
#include "utils/logger.h"
#include "utils/timer.h"
#include "utils/util_functions.h"
#include <random>

using namespace Aperture;

int
main(int argc, char* argv[]) {
  uint32_t step = 0;
  // Construct the simulation environment
  // cu_sim_environment env(&argc, &argv);

  // Allocate simulation data
  // cu_sim_data data(env);

  // Initialize the field solver
  // field_solver_default solver(env.grid());

  // Initialize data exporter
  // data_exporter exporter(env.params(), step);
  return 0;
}
