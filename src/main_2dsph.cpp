#include "algorithms/field_solver_log_sph.h"
#include "sim_data.h"
#include "sim_environment.h"
#include "utils/logger.h"
#include "utils/timer.h"
#include <random>

using namespace Aperture;

int main(int argc, char *argv[])
{
  // Construct the simulation environment
  Environment env(&argc, &argv);

  // Allocate simulation data
  SimData data(env);

  // Initialize the field solver
  FieldSolver_LogSph field_solver(*dynamic_cast<const Grid_LogSph*>(&env.local_grid()));

  DataExporter exporter(env.params(), "/home/alex/storage/Data/Aperture3/2d_test/",
                        "data", 1);
  exporter.WriteGrid();

  data.E.initialize();
  data.B.initialize();
  exporter.AddField("E", data.E);
  exporter.AddField("B", data.B);

  exporter.WriteOutput(0, 0.0);
  exporter.writeXMF(0, 0.0);
  return 0;
}
