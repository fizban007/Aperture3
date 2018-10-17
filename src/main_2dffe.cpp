#include "algorithms/field_solver_ffe_cyl.h"
#include "sim_data.h"
#include "sim_environment.h"
#include "utils/logger.h"
#include "utils/timer.h"
#include "utils/util_functions.h"
#include <fmt/core.h>
#include <fstream>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>
#include <iostream>
#include <random>
#include <vector>

using namespace Aperture;
using namespace HighFive;

void
compute_flux(ScalarField<Scalar>& flux, VectorField<Scalar>& B) {
  B.sync_to_host(1);
  flux.initialize();
  auto& mesh = B.grid().mesh();
  for (int j = mesh.delta[1]; j < mesh.dims[1] - mesh.delta[1]; j++) {
    for (int i = mesh.delta[0]; i < mesh.dims[0] - mesh.delta[0]; i++) {
      Scalar r = mesh.pos(0, i, true);
      flux(i, j) = flux(i - 1, j) + mesh.delta[0] * r * B(1, i, j);
    }
  }
  // flux.sync_to_device();
}

int
main(int argc, char* argv[]) {
  Environment env(&argc, &argv);

  // Print the parameters of this run
  Logger::print_info("dt is {}", env.params().delta_t);

  // Allocate simulation data
  SimData data(env);

  // Initialize the field solver
  FieldSolver_FFE_Cyl field_solver(env.grid());

  // TODO: Initialize the fields

  // Define the flux field
  ScalarField<Scalar> flux(env.grid());

  // Run the simulation
  for (unsigned int step = 0; step < env.params().max_steps; step++) {
    Logger::print_info("At timestep {}", step);
    timer::stamp();
    field_solver.update_fields(data, env.params().delta_t);
    timer::show_duration_since_stamp("FFE step", "ms");


    if (step % env.params().data_interval == 0) {
      compute_flux(flux, data.B);
    }
  }

  return 0;
}
