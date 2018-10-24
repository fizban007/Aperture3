#include "algorithms/field_solver_ffe_cyl.h"
#include "sim_data.h"
#include "sim_environment.h"
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
compute_flux(ScalarField<Scalar>& flux, VectorField<Scalar>& B) {
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
  Environment env(&argc, &argv);

  // Print the parameters of this run
  Logger::print_info("dt is {}", env.params().delta_t);
  auto& mesh = env.grid().mesh();

  // Allocate simulation data
  SimData data(env);

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

  DataExporter exporter(env.grid(), "./ffe_cyl/", "data", 1);
  exporter.WriteGrid();

  // ScalarField<Scalar> x1(env.grid()), x2(env.grid());
  // x1.initialize([](Scalar x1, Scalar x2, Scalar x3){
  //                 return x1;
  //               });
  // x2.initialize([](Scalar x1, Scalar x2, Scalar x3){
  //                 return x2;
  //               });
  // Scalar** x1_ptr = new Scalar*[mesh.dims[1]];
  // Scalar** x2_ptr = new Scalar*[mesh.dims[1]];
  // boost::multi_array<Scalar, 2> x1_array(
  //     boost::extents[mesh.dims[1]][mesh.dims[0]]);
  // boost::multi_array<Scalar, 2> x2_array(
  //     boost::extents[mesh.dims[1]][mesh.dims[0]]);
  // boost::multi_array<Scalar, 2> x1_array(
  //     boost::extents[mesh.dims[1]][mesh.dims[0]]);
  // boost::multi_array<Scalar, 2> x2_array(
  //     boost::extents[mesh.dims[1]][mesh.dims[0]]);
  // for (int i = 0; i < 100; i++) {
  //   Logger::print_info("{}", data.B(0, i, 100));
  // }

  // Define the flux field
  ScalarField<Scalar> flux(env.grid());

  // Define the dynamic 2D array of the data, a hack
  Scalar** flux_ptr = new Scalar*[mesh.dims[1]];
  Scalar** b1_ptr = new Scalar*[mesh.dims[1]];
  Scalar** b2_ptr = new Scalar*[mesh.dims[1]];
  Scalar** bphi_ptr = new Scalar*[mesh.dims[1]];
  Scalar** e1_ptr = new Scalar*[mesh.dims[1]];
  Scalar** e2_ptr = new Scalar*[mesh.dims[1]];
  Scalar** e3_ptr = new Scalar*[mesh.dims[1]];
  for (int i = 0; i < mesh.dims[1]; i++) {
    flux_ptr[i] = flux.data().data() + i * mesh.dims[0];
    bphi_ptr[i] = data.B.data(2).data() + i * mesh.dims[0];
    b1_ptr[i] = data.B.data(0).data() + i * mesh.dims[0];
    b2_ptr[i] = data.B.data(1).data() + i * mesh.dims[0];

    e1_ptr[i] = data.E.data(0).data() + i * mesh.dims[0];
    e2_ptr[i] = data.E.data(1).data() + i * mesh.dims[0];
    e3_ptr[i] = data.E.data(2).data() + i * mesh.dims[0];

    // x1_ptr[i] = x1.data().data() + i * mesh.dims[0];
    // x2_ptr[i] = x2.data().data() + i * mesh.dims[0];
    // Logger::print_info("{}", b2_ptr[i][100]);
  }

  // std::string meshfile_name = "./ffe_cyl/mesh.h5";
  // File meshfile(meshfile_name.c_str(),
  //               File::ReadWrite | File::Create | File::Truncate);
  std::vector<size_t> dims(2);
  dims[0] = env.grid().mesh().dims[0];
  dims[1] = env.grid().mesh().dims[1];
  // DataSet mesh_x1 =
  //     meshfile.createDataSet<Scalar>("x1",
  //     DataSpace::From(x1_array));
  // mesh_x1.write(x1_array);
  // DataSet mesh_x2 =
  //     meshfile.createDataSet<Scalar>("x2",
  //     DataSpace::From(x2_array));
  // mesh_x2.write(x2_array);
  // delete[] x1_ptr;
  // delete[] x2_ptr;
  exporter.AddField("B", data.B);
  exporter.AddField("E", data.E);
  exporter.AddField("flux", flux);

  // std::ofstream fs("./ffe_cyl/data_test.xmf");
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
      Logger::print_info("Export data here");
      compute_flux(flux, data.B);
      std::string filename =
          fmt::format("./ffe_cyl/data{:06d}.h5", step);
      Logger::print_info("file name is {}", filename);
      File datafile(filename.c_str(),
                    File::ReadWrite | File::Create | File::Truncate);

      // Create the dataset
      DataSet data_flux =
          datafile.createDataSet<Scalar>("flux", DataSpace(dims));
      data_flux.write((Scalar**)flux_ptr[0]);
      DataSet data_bphi =
          datafile.createDataSet<Scalar>("Bphi", DataSpace(dims));
      data_bphi.write((Scalar**)bphi_ptr[0]);
      DataSet data_b1 =
          datafile.createDataSet<Scalar>("B1", DataSpace(dims));
      data_b1.write((Scalar**)b1_ptr[0]);
      DataSet data_b2 =
          datafile.createDataSet<Scalar>("B2", DataSpace(dims));
      data_b2.write((Scalar**)b2_ptr[0]);
      DataSet data_e1 =
          datafile.createDataSet<Scalar>("E1", DataSpace(dims));
      data_e1.write((Scalar**)e1_ptr[0]);
      DataSet data_e2 =
          datafile.createDataSet<Scalar>("E2", DataSpace(dims));
      data_e2.write((Scalar**)e2_ptr[0]);
      DataSet data_e3 =
          datafile.createDataSet<Scalar>("E3", DataSpace(dims));
      data_e3.write((Scalar**)e3_ptr[0]);

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
  // fs.close();

  delete[] flux_ptr;
  delete[] bphi_ptr;
  delete[] b1_ptr;
  delete[] b2_ptr;
  delete[] e1_ptr;
  delete[] e2_ptr;
  delete[] e3_ptr;

  return 0;
}
