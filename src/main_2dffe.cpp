#include "algorithms/field_solver_ffe_cyl.h"
#include "cu_sim_data.h"
#include "cu_sim_environment.h"
#include "utils/hdf_exporter.h"
#include "utils/logger.h"
#include "utils/timer.h"
#include "utils/util_functions.h"
#include <fmt/core.h>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

#define H5_USE_BOOST

#include <boost/multi_array.hpp>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>

using namespace Aperture;
using namespace HighFive;

void
compute_flux(cu_scalar_field<Scalar>& flux, cu_vector_field<Scalar>& B) {
  // B.sync_to_host(1);
  flux.initialize();
  flux.sync_to_host();
  auto& mesh = B.grid().mesh();
  for (int j = mesh.guard[1]; j < mesh.dims[1] - mesh.guard[1]; j++) {
    for (int i = mesh.guard[0]; i < mesh.dims[0] - mesh.guard[0]; i++) {
      Scalar r = mesh.pos(0, i, true);
      flux(i, j) = flux(i - 1, j) + mesh.delta[0] * r * B(1, i, j);
      // if (j == 100)
      //   Logger::print_info("{}, {}, {}", r, B(1, i, j), flux(i, j));
    }
  }
  // flux.sync_to_device();
}

int
main(int argc, char* argv[]) {
  cu_sim_environment env(&argc, &argv);

  // Print the parameters of this run
  Logger::print_info("dt is {}", env.params().delta_t);
  auto& mesh = env.grid().mesh();

  // Allocate simulation data
  cu_sim_data data(env);

  // Initialize the field solver
  FieldSolver_FFE_Cyl field_solver(env.grid());

  // TODO: Initialize the fields
  Scalar B0 = env.params().B0;
  data.E.initialize();
  data.B.initialize(0, [B0](Scalar x1, Scalar x2, Scalar x3) {
    return 1.5f * B0 * x1 * x2 / std::pow(x1 * x1 + x2 * x2, 2.5f);
  });
  data.B.initialize(1, [B0](Scalar x1, Scalar x2, Scalar x3) {
    return (B0 * x2 * x2 - 0.5f * B0 * x1 * x1) /
           std::pow(x1 * x1 + x2 * x2, 2.5f);
  });
  data.B.initialize(
      2, [](Scalar x1, Scalar x2, Scalar x3) { return 0.0f; });
  data.B.sync_to_device();

  DataExporter exporter(env.params(), "./ffe_cyl/", "data", 2);
  exporter.WriteGrid();

  cu_scalar_field<Scalar> flux(env.grid());

  exporter.AddField("B", data.B);
  exporter.AddField("E", data.E);
  exporter.AddField("flux", flux);

  // Run the simulation
  for (unsigned int step = 0; step < env.params().max_steps; step++) {
    Scalar time = env.params().delta_t * step;
    Scalar omega = 0.0;
    if (time < 0.1)
      omega = 100.f * time;
    else if (time < 0.2)
      omega = 100.f * (0.2 - time);

    if ((step % env.params().data_interval) == 0) {
      data.B.sync_to_host();
      data.E.sync_to_host();

      Logger::print_info("Export data here");
      compute_flux(flux, data.B);

      exporter.WriteOutput(step, time);
      exporter.writeXMF(step, time);
    }

    Logger::print_info("At timestep {}, omega is {}", step, omega);
    timer::stamp();
    field_solver.update_fields(data, env.params().delta_t, omega);
    // field_solver.update_field_substep(data.E, data.B, data.J,
    //                                   data.E, data.B,
    //                                   env.params().delta_t);
    timer::show_duration_since_stamp("FFE step", "ms");
  }


  return 0;
}
