#include "utils/hdf_exporter.h"
#include "fmt/core.h"
#include "utils/logger.h"
// #include "config_file.h"
#include "commandline_args.h"
#include "nlohmann/json.hpp"
#include "sim_params.h"
// #include <H5Cpp.h>
#include <boost/filesystem.hpp>
#include <fstream>

#define H5_USE_BOOST

#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>
#include <time.h>

using json = nlohmann::json;
using namespace HighFive;

namespace Aperture {

// DataExporter::DataExporter() {}

DataExporter::DataExporter(const SimParams &params,
                           const std::string &dir,
                           const std::string &prefix, int downsample)
    : outputDirectory(dir),
      filePrefix(prefix),
      downsample_factor(downsample) {
  boost::filesystem::path rootPath(dir.c_str());
  boost::system::error_code returnedError;

  boost::filesystem::create_directories(rootPath, returnedError);
  if (outputDirectory.back() != '/') outputDirectory.push_back('/');

  grid.init(params);
  for (int i = 0; i < grid.dim(); i++) {
    grid.mesh().dims[i] /= params.N[i] / downsample + 2 * params.guard[i];
    grid.mesh().delta[i] *= downsample;
    grid.mesh().inv_delta[i] /= downsample;
  }
  // Format the output directory as Data%Y%m%d-%H%M
  // char myTime[150] = {};
  // char subDir[200] = {};
  // time_t rawtime;
  // struct tm *timeinfo;
  // time(&rawtime);
  // timeinfo = localtime(&rawtime);
  // strftime(myTime, 140, "%Y%m%d-%H%M", timeinfo);
  // snprintf(subDir, sizeof(subDir), "Data%s/", myTime);

  // outputDirectory += subDir;
}

DataExporter::~DataExporter() {}

void
DataExporter::createDirectories() {
  boost::filesystem::path subPath(outputDirectory);
  boost::filesystem::path logPath(outputDirectory + "log/");

  boost::system::error_code returnedError;
  boost::filesystem::create_directories(subPath, returnedError);
  boost::filesystem::create_directories(logPath, returnedError);
}

bool
DataExporter::checkDirectories() {
  boost::filesystem::path subPath(outputDirectory);

  return boost::filesystem::exists(outputDirectory);
}

// void
// DataExporter::AddArray(const std::string &name, float *data, int
// *dims,
//                        int ndims) {
//   dataset<float> tempData;

//   tempData.name = name;
//   tempData.data = data;
//   tempData.ndims = ndims;
//   for (int i = 0; i < ndims; i++) tempData.dims.push_back(dims[i]);

//   dbFloat.push_back(tempData);
// }

// void
// DataExporter::AddArray(const std::string &name, double *data, int
// *dims,
//                        int ndims) {
//   dataset<double> tempData;

//   tempData.name = name;
//   tempData.data = data;
//   tempData.ndims = ndims;
//   for (int i = 0; i < ndims; i++) tempData.dims.push_back(dims[i]);

//   dbDouble.push_back(tempData);
// }

// template <typename T>
// void
// DataExporter::AddArray(const std::string &name, MultiArray<T> &array)
// {
//   int ndims = array.dim();
//   int *dims = new int[ndims];
//   for (int i = 0; i < ndims; i++) dims[i] = array.extent()[i];

//   AddArray(name, array.data(), dims, ndims);

//   delete[] dims;
// }

// template <typename T>
// void
// DataExporter::AddArray(const std::string &name, VectorField<T>
// &field,
//                        int component) {
//   AddArray(name, field.data(component));
// }

template <typename T>
void
DataExporter::AddField(const std::string &name,
                       const ScalarField<T> &field) {
  auto &mesh = grid.mesh();

  if (grid.dim() == 3) {
    sfieldoutput3d<T> tempData;
    tempData.name = name;
    tempData.field = &field;

    tempData.f.resize(
        boost::extents[mesh.dims[2]][mesh.dims[1]][mesh.dims[0]]);
    dbScalars3d.push_back(std::move(tempData));
  } else if (grid.dim() == 2) {
    sfieldoutput2d<T> tempData;
    tempData.name = name;
    tempData.field = &field;

    tempData.f.resize(boost::extents[mesh.dims[1]][mesh.dims[0]]);
    dbScalars2d.push_back(std::move(tempData));
  }
}

template <typename T>
void
DataExporter::AddField(const std::string &name,
                       const VectorField<T> &field) {
  auto &mesh = grid.mesh();

  if (grid.dim() == 3) {
    vfieldoutput3d<T> tempData;
    tempData.name = name;
    tempData.field = &field;
    tempData.f1.resize(
        boost::extents[mesh.dims[2]][mesh.dims[1]][mesh.dims[0]]);
    tempData.f2.resize(
        boost::extents[mesh.dims[2]][mesh.dims[1]][mesh.dims[0]]);
    tempData.f3.resize(
        boost::extents[mesh.dims[2]][mesh.dims[1]][mesh.dims[0]]);
    dbVectors3d.push_back(std::move(tempData));
  } else if (grid.dim() == 2) {
    vfieldoutput2d<T> tempData;
    tempData.name = name;
    tempData.field = &field;
    tempData.f1.resize(boost::extents[mesh.dims[1]][mesh.dims[0]]);
    tempData.f2.resize(boost::extents[mesh.dims[1]][mesh.dims[0]]);
    tempData.f3.resize(boost::extents[mesh.dims[1]][mesh.dims[0]]);
    dbVectors2d.push_back(std::move(tempData));
  }
}

void
DataExporter::InterpolateFieldValues() {
  for (auto &sf : dbScalars3d) {
    auto &mesh = sf.field->grid().mesh();
    for (int k = 0; k < mesh.reduced_dim(2); k += downsample_factor) {
      for (int j = 0; j < mesh.reduced_dim(1); j += downsample_factor) {
        for (int i = 0; i < mesh.reduced_dim(0);
             i += downsample_factor) {
          sf.f[k / downsample_factor + mesh.guard[2]]
              [j / downsample_factor + mesh.guard[1]]
              [i / downsample_factor + mesh.guard[0]] = (*sf.field)(
              i + mesh.guard[0], j + mesh.guard[1], k + mesh.guard[2]);
        }
      }
    }
  }

  for (auto &vf : dbVectors3d) {
    auto &mesh = vf.field->grid().mesh();
    for (int k = 0; k < mesh.reduced_dim(2); k += downsample_factor) {
      for (int j = 0; j < mesh.reduced_dim(1); j += downsample_factor) {
        for (int i = 0; i < mesh.reduced_dim(0);
             i += downsample_factor) {
          vf.f1[k / downsample_factor + mesh.guard[2]]
               [j / downsample_factor + mesh.guard[1]]
               [i / downsample_factor + mesh.guard[0]] =
              (*vf.field)(0, i + mesh.guard[0], j + mesh.guard[1],
                          k + mesh.guard[2]);
          vf.f2[k / downsample_factor + mesh.guard[2]]
               [j / downsample_factor + mesh.guard[1]]
               [i / downsample_factor + mesh.guard[0]] =
              (*vf.field)(1, i + mesh.guard[0], j + mesh.guard[1],
                          k + mesh.guard[2]);
          vf.f3[k / downsample_factor + mesh.guard[2]]
               [j / downsample_factor + mesh.guard[1]]
               [i / downsample_factor + mesh.guard[0]] =
              (*vf.field)(2, i + mesh.guard[0], j + mesh.guard[1],
                          k + mesh.guard[2]);
        }
      }
    }
  }

  for (auto &sf : dbScalars2d) {
    auto &mesh = sf.field->grid().mesh();
    for (int j = 0; j < mesh.reduced_dim(1); j += downsample_factor) {
      for (int i = 0; i < mesh.reduced_dim(0); i += downsample_factor) {
        sf.f[j / downsample_factor + mesh.guard[1]]
            [i / downsample_factor + mesh.guard[0]] =
            (*sf.field)(i + mesh.guard[0], j + mesh.guard[1]);
      }
    }
  }

  for (auto &vf : dbVectors2d) {
    Logger::print_info("Writing {}", vf.name);
    auto &mesh = vf.field->grid().mesh();
    for (int j = 0; j < mesh.reduced_dim(1); j += downsample_factor) {
      for (int i = 0; i < mesh.reduced_dim(0); i += downsample_factor) {
        vf.f1[j / downsample_factor + mesh.guard[1]]
             [i / downsample_factor + mesh.guard[0]] =
            (*vf.field)(0, i + mesh.guard[0], j + mesh.guard[1]);
        // std::cout << vf.f1[j / downsample_factor + mesh.guard[1]]
        //     [i / downsample_factor + mesh.guard[0]] << std::endl;
        vf.f2[j / downsample_factor + mesh.guard[1]]
             [i / downsample_factor + mesh.guard[0]] =
            (*vf.field)(1, i + mesh.guard[0], j + mesh.guard[1]);
        vf.f3[j / downsample_factor + mesh.guard[1]]
             [i / downsample_factor + mesh.guard[0]] =
            (*vf.field)(2, i + mesh.guard[0], j + mesh.guard[1]);
      }
    }
  }
}

void
DataExporter::AddParticleArray(const std::string &name,
                               const Particles &ptc) {
  ptcoutput<Particles> temp;
  temp.name = name;
  temp.ptc = &ptc;
  temp.data_x = std::vector<float>(MAX_TRACKED);
  temp.data_p = std::vector<float>(MAX_TRACKED);

  dbPtcData.push_back(std::move(temp));
}

void
DataExporter::AddParticleArray(const std::string &name,
                               const Photons &ptc) {
  ptcoutput<Photons> temp;
  temp.name = name;
  temp.ptc = &ptc;
  temp.data_x = std::vector<float>(MAX_TRACKED);
  temp.data_p = std::vector<float>(MAX_TRACKED);
  temp.data_l = std::vector<float>(MAX_TRACKED);

  dbPhotonData.push_back(std::move(temp));
}

void
DataExporter::WriteGrid() {
  if (grid.dim() == 3) {
    auto &mesh = grid.mesh();
    boost::multi_array<float, 3> x1_array(
        boost::extents[mesh.dims[2]][mesh.dims[1]][mesh.dims[0]]);
    boost::multi_array<float, 3> x2_array(
        boost::extents[mesh.dims[2]][mesh.dims[1]][mesh.dims[0]]);
    boost::multi_array<float, 3> x3_array(
        boost::extents[mesh.dims[2]][mesh.dims[1]][mesh.dims[0]]);

    for (int k = 0; k < mesh.dims[2]; k++) {
      for (int j = 0; j < mesh.dims[1]; j++) {
        for (int i = 0; i < mesh.dims[0]; i++) {
          x1_array[k][j][i] = mesh.pos(0, i, false);
          x2_array[k][j][i] = mesh.pos(1, j, false);
          x3_array[k][j][i] = mesh.pos(2, k, false);
        }
      }
    }

    std::string meshfilename = outputDirectory + "mesh.h5";
    Logger::print_info("{}", meshfilename);
    File meshfile(meshfilename.c_str(),
                  File::ReadWrite | File::Create | File::Truncate);
    DataSet mesh_x1 =
        meshfile.createDataSet<float>("x1", DataSpace::From(x1_array));
    mesh_x1.write(x1_array);
    DataSet mesh_x2 =
        meshfile.createDataSet<float>("x2", DataSpace::From(x2_array));
    mesh_x2.write(x2_array);
    DataSet mesh_x3 =
        meshfile.createDataSet<float>("x3", DataSpace::From(x3_array));
    mesh_x3.write(x3_array);
  } else if (grid.dim() == 2) {
    auto &mesh = grid.mesh();
    boost::multi_array<float, 2> x1_array(
        boost::extents[mesh.dims[1]][mesh.dims[0]]);
    boost::multi_array<float, 2> x2_array(
        boost::extents[mesh.dims[1]][mesh.dims[0]]);

    for (int j = 0; j < mesh.dims[1]; j++) {
      for (int i = 0; i < mesh.dims[0]; i++) {
        x1_array[j][i] = mesh.pos(0, i, false);
        x2_array[j][i] = mesh.pos(1, j, false);
      }
    }

    std::string meshfilename = outputDirectory + "mesh.h5";
    Logger::print_info("{}", meshfilename);
    File meshfile(meshfilename.c_str(),
                  File::ReadWrite | File::Create | File::Truncate);
    DataSet mesh_x1 =
        meshfile.createDataSet<float>("x1", DataSpace::From(x1_array));
    mesh_x1.write(x1_array);
    DataSet mesh_x2 =
        meshfile.createDataSet<float>("x2", DataSpace::From(x2_array));
    mesh_x2.write(x2_array);
  }
}

void
DataExporter::WriteOutput(int timestep, double time) {
  if (!checkDirectories()) createDirectories();
  InterpolateFieldValues();
  File datafile(fmt::format("{}{}{:06d}.h5", outputDirectory,
                            filePrefix, timestep)
                    .c_str(),
                File::ReadWrite | File::Create | File::Truncate);
  // try {
  //   std::string filename = outputDirectory + filePrefix +
  //                          fmt::format("{0:06d}.h5", timestep);
  //   H5::H5File *file = new H5::H5File(filename, H5F_ACC_TRUNC);

  //   for (auto &ds : dbFloat) {
  //     hsize_t *sizes = new hsize_t[ds.ndims];
  //     for (int i = 0; i < ds.ndims; i++) sizes[i] = ds.dims[i];
  //     H5::DataSpace space(ds.ndims, sizes);
  //     H5::DataSet *dataset = new H5::DataSet(file->createDataSet(
  //         ds.name, H5::PredType::NATIVE_FLOAT, space));
  //     dataset->write((void *)ds.data, H5::PredType::NATIVE_FLOAT);

  //     delete[] sizes;
  //     delete dataset;
  //   }

  //   for (auto &ds : dbDouble) {
  //     hsize_t *sizes = new hsize_t[ds.ndims];
  //     for (int i = 0; i < ds.ndims; i++) sizes[i] = ds.dims[i];
  //     H5::DataSpace space(ds.ndims, sizes);
  //     H5::DataSet *dataset = new H5::DataSet(file->createDataSet(
  //         ds.name, H5::PredType::NATIVE_DOUBLE, space));
  //     dataset->write((void *)ds.data, H5::PredType::NATIVE_DOUBLE);

  //     delete[] sizes;
  //     delete dataset;
  //   }

  //   for (auto &ds : dbPtcData) {
  //     std::string name_x = ds.name + "_x";
  //     std::string name_p = ds.name + "_p";
  //     unsigned int idx = 0;
  //     for (Index_t n = 0; n < ds.ptc->number(); n++) {
  //       if (!ds.ptc->is_empty(n) &&
  //           ds.ptc->check_flag(n, ParticleFlag::tracked) &&
  //           idx < MAX_TRACKED) {
  //         Scalar x = grid.mesh().pos(0, ds.ptc->data().cell[n],
  //                                    ds.ptc->data().x1[n]);
  //         ds.data_x[idx] = x;
  //         ds.data_p[idx] = ds.ptc->data().p1[n];
  //         idx += 1;
  //       }
  //     }
  //     hsize_t sizes[1] = {idx};
  //     H5::DataSpace space(1, sizes);
  //     H5::DataSet *dataset_x = new H5::DataSet(file->createDataSet(
  //         name_x, H5::PredType::NATIVE_FLOAT, space));
  //     dataset_x->write((void *)ds.data_x.data(),
  //                      H5::PredType::NATIVE_FLOAT);
  //     H5::DataSet *dataset_p = new H5::DataSet(file->createDataSet(
  //         name_p, H5::PredType::NATIVE_FLOAT, space));
  //     dataset_p->write((void *)ds.data_p.data(),
  //                      H5::PredType::NATIVE_FLOAT);

  //     delete dataset_x;
  //     delete dataset_p;

  //     Logger::print_info("Written {} tracked particles", idx);
  //   }

  //   for (auto &ds : dbPhotonData) {
  //     std::string name_x = ds.name + "_x";
  //     std::string name_p = ds.name + "_p";
  //     // std::string name_l = ds.name + "_l";
  //     unsigned int idx = 0;
  //     for (Index_t n = 0; n < ds.ptc->number(); n++) {
  //       if (!ds.ptc->is_empty(n) &&
  //           ds.ptc->check_flag(n, PhotonFlag::tracked) &&
  //           idx < MAX_TRACKED) {
  //         Scalar x = grid.mesh().pos(0, ds.ptc->data().cell[n],
  //                                    ds.ptc->data().x1[n]);
  //         ds.data_x[idx] = x;
  //         ds.data_p[idx] = ds.ptc->data().p1[n];
  //         // ds.data_l[idx] = ds.ptc->data().path[n];
  //         idx += 1;
  //       }
  //     }
  //     hsize_t sizes[1] = {idx};
  //     H5::DataSpace space(1, sizes);
  //     H5::DataSet *dataset_x = new H5::DataSet(file->createDataSet(
  //         name_x, H5::PredType::NATIVE_FLOAT, space));
  //     dataset_x->write((void *)ds.data_x.data(),
  //                      H5::PredType::NATIVE_FLOAT);
  //     H5::DataSet *dataset_p = new H5::DataSet(file->createDataSet(
  //         name_p, H5::PredType::NATIVE_FLOAT, space));
  //     dataset_p->write((void *)ds.data_p.data(),
  //                      H5::PredType::NATIVE_FLOAT);
  //     // H5::DataSet *dataset_l = new
  //     // H5::DataSet(file->createDataSet(name_l,
  //     // H5::PredType::NATIVE_FLOAT, space));
  //     // dataset_l->write((void*)ds.data_l.data(),
  //     // H5::PredType::NATIVE_FLOAT);

  //     delete dataset_x;
  //     delete dataset_p;
  //     // delete dataset_l;

  //     Logger::print_info("Written {} tracked photons", idx);
  //   }
  //   delete file;
  // }
  // // catch failure caused by the H5File operations
  // catch (H5::FileIException &error) {
  //   error.printErrorStack();
  //   // return -1;
  // }
  // // catch failure caused by the DataSet operations
  // catch (H5::DataSetIException &error) {
  //   error.printErrorStack();
  //   // return -1;
  // }
  // // catch failure caused by the DataSpace operations
  // catch (H5::DataSpaceIException &error) {
  //   error.printErrorStack();
  //   // return -1;
  // }
  for (auto &sf : dbScalars2d) {
    DataSet data =
        datafile.createDataSet<float>(sf.name, DataSpace::From(sf.f));
    data.write(sf.f);
  }

  for (auto &sf : dbScalars3d) {
    DataSet data =
        datafile.createDataSet<float>(sf.name, DataSpace::From(sf.f));
    data.write(sf.f);
  }

  for (auto &vf : dbVectors2d) {
    DataSet data1 = datafile.createDataSet<float>(
        fmt::format("{}1", vf.name), DataSpace::From(vf.f1));
    data1.write(vf.f1);
    DataSet data2 = datafile.createDataSet<float>(
        fmt::format("{}2", vf.name), DataSpace::From(vf.f2));
    data2.write(vf.f2);
    DataSet data3 = datafile.createDataSet<float>(
        fmt::format("{}3", vf.name), DataSpace::From(vf.f3));
    data3.write(vf.f3);
  }

  for (auto &vf : dbVectors3d) {
    DataSet data1 = datafile.createDataSet<float>(
        fmt::format("{}1", vf.name), DataSpace::From(vf.f1));
    data1.write(vf.f1);
    DataSet data2 = datafile.createDataSet<float>(
        fmt::format("{}2", vf.name), DataSpace::From(vf.f2));
    data2.write(vf.f2);
    DataSet data3 = datafile.createDataSet<float>(
        fmt::format("{}3", vf.name), DataSpace::From(vf.f3));
    data3.write(vf.f3);
  }
}

void
DataExporter::writeConfig(const SimParams &params) {
  if (!checkDirectories()) createDirectories();
  std::string filename = outputDirectory + "config.json";
  auto &c = params;
  json conf = {{"delta_t", c.delta_t},
               {"q_e", c.q_e},
               {"ptc_per_cell", c.ptc_per_cell},
               {"ion_mass", c.ion_mass},
               {"num_species", c.num_species},
               {"periodic_boundary", c.periodic_boundary},
               {"create_pairs", c.create_pairs},
               {"trace_photons", c.trace_photons},
               {"gamma_thr", c.gamma_thr},
               {"photon_path", c.photon_path},
               {"ic_path", c.ic_path},
               {"rad_energy_bins", c.rad_energy_bins},
               // {"grid", {
               {"N", c.N},
               {"guard", c.guard},
               {"lower", c.lower},
               {"size", c.size},
               //   }},
               {"interp_order", c.interpolation_order},
               {"track_pct", c.track_percent},
               {"N_steps", c.max_steps},
               {"data_interval", c.data_interval},
               {"spectral_alpha", c.spectral_alpha},
               {"e_s", c.e_s},
               {"e_min", c.e_min}};

  std::ofstream o(filename);
  o << std::setw(4) << conf << std::endl;
}

void
DataExporter::writeXMFHead(std::ofstream &fs) {
  fs << "<?xml version=\"1.0\" ?>" << std::endl;
  fs << "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>" << std::endl;
  fs << "<Xdmf>" << std::endl;
  fs << "<Domain>" << std::endl;
  fs << "<Grid Name=\"Aperture\" GridType=\"Collection\" "
        "CollectionType=\"Temporal\" >"
     << std::endl;
}

void
DataExporter::writeXMFStep(std::ofstream &fs, int step, double time) {
  std::string dim_str;
  auto &mesh = grid.mesh();
  if (grid.dim() == 3) {
    dim_str = fmt::format("{} {} {}", mesh.dims[2], mesh.dims[1],
                          mesh.dims[0]);
  } else if (grid.dim() == 2) {
    dim_str = fmt::format("{} {}", mesh.dims[1], mesh.dims[0]);
  }

  fs << "<Grid Name=\"Aperture\" Type=\"Uniform\">" << std::endl;
  fs << "  <Time Type=\"Single\" Value=\"" << time << "\"/>"
     << std::endl;
  if (grid.dim() == 3) {
    fs << "  <Topology Type=\"3DSMesh\" NumberOfElements=\"" << dim_str
       << "\"/>" << std::endl;
    fs << "  <Geometry GeometryType=\"X_Y_Z\">" << std::endl;
  } else if (grid.dim() == 2) {
    fs << "  <Topology Type=\"2DSMesh\" NumberOfElements=\"" << dim_str
       << "\"/>" << std::endl;
    fs << "  <Geometry GeometryType=\"X_Y\">" << std::endl;
  }
  fs << "    <DataItem Dimensions=\"" << dim_str
     << "\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">"
     << std::endl;
  fs << "      mesh.h5:x1" << std::endl;
  fs << "    </DataItem>" << std::endl;
  fs << "    <DataItem Dimensions=\"" << dim_str
     << "\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">"
     << std::endl;
  fs << "      mesh.h5:x2" << std::endl;
  fs << "    </DataItem>" << std::endl;
  if (grid.dim() == 3) {
    fs << "    <DataItem Dimensions=\"" << dim_str
       << "\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">"
       << std::endl;
    fs << "      mesh.h5:x3" << std::endl;
    fs << "    </DataItem>" << std::endl;
  }

  fs << "  </Geometry>" << std::endl;

  for (auto &sf : dbScalars2d) {
    fs << "  <Attribute Name=\"" << sf.name
       << "\" Center=\"Node\" AttributeType=\"Scalar\">" << std::endl;
    fs << "    <DataItem Dimensions=\"" << dim_str
       << "\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">"
       << std::endl;
    fs << fmt::format("      {}{:06d}.h5:{}", filePrefix, step, sf.name)
       << std::endl;
    fs << "    </DataItem>" << std::endl;
    fs << "  </Attribute>" << std::endl;
  }
  for (auto &sf : dbScalars3d) {
    fs << "  <Attribute Name=\"" << sf.name
       << "\" Center=\"Node\" AttributeType=\"Scalar\">" << std::endl;
    fs << "    <DataItem Dimensions=\"" << dim_str
       << "\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">"
       << std::endl;
    fs << fmt::format("      {}{:06d}.h5:{}", filePrefix, step, sf.name)
       << std::endl;
    fs << "    </DataItem>" << std::endl;
    fs << "  </Attribute>" << std::endl;
  }
  for (auto &vf : dbVectors2d) {
    fs << "  <Attribute Name=\"" << vf.name
       << "1\" Center=\"Node\" AttributeType=\"Scalar\">" << std::endl;
    fs << "    <DataItem Dimensions=\"" << dim_str
       << "\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">"
       << std::endl;
    fs << fmt::format("      {}{:06d}.h5:{}1", filePrefix, step,
                      vf.name)
       << std::endl;
    fs << "    </DataItem>" << std::endl;
    fs << "  </Attribute>" << std::endl;
    fs << "  <Attribute Name=\"" << vf.name
       << "2\" Center=\"Node\" AttributeType=\"Scalar\">" << std::endl;
    fs << "    <DataItem Dimensions=\"" << dim_str
       << "\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">"
       << std::endl;
    fs << fmt::format("      {}{:06d}.h5:{}2", filePrefix, step,
                      vf.name)
       << std::endl;
    fs << "    </DataItem>" << std::endl;
    fs << "  </Attribute>" << std::endl;
    fs << "  <Attribute Name=\"" << vf.name
       << "3\" Center=\"Node\" AttributeType=\"Scalar\">" << std::endl;
    fs << "    <DataItem Dimensions=\"" << dim_str
       << "\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">"
       << std::endl;
    fs << fmt::format("      {}{:06d}.h5:{}3", filePrefix, step,
                      vf.name)
       << std::endl;
    fs << "    </DataItem>" << std::endl;
    fs << "  </Attribute>" << std::endl;
  }

  for (auto &vf : dbVectors3d) {
    fs << "  <Attribute Name=\"" << vf.name
       << "1\" Center=\"Node\" AttributeType=\"Scalar\">" << std::endl;
    fs << "    <DataItem Dimensions=\"" << dim_str
       << "\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">"
       << std::endl;
    fs << fmt::format("      {}{:06d}.h5:{}1", filePrefix, step,
                      vf.name)
       << std::endl;
    fs << "    </DataItem>" << std::endl;
    fs << "  </Attribute>" << std::endl;
    fs << "  <Attribute Name=\"" << vf.name
       << "2\" Center=\"Node\" AttributeType=\"Scalar\">" << std::endl;
    fs << "    <DataItem Dimensions=\"" << dim_str
       << "\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">"
       << std::endl;
    fs << fmt::format("      {}{:06d}.h5:{}2", filePrefix, step,
                      vf.name)
       << std::endl;
    fs << "    </DataItem>" << std::endl;
    fs << "  </Attribute>" << std::endl;
    fs << "  <Attribute Name=\"" << vf.name
       << "3\" Center=\"Node\" AttributeType=\"Scalar\">" << std::endl;
    fs << "    <DataItem Dimensions=\"" << dim_str
       << "\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">"
       << std::endl;
    fs << fmt::format("      {}{:06d}.h5:{}3", filePrefix, step,
                      vf.name)
       << std::endl;
    fs << "    </DataItem>" << std::endl;
    fs << "  </Attribute>" << std::endl;
  }

  fs << "</Grid>" << std::endl;
}

void
DataExporter::writeXMFTail(std::ofstream &fs) {
  fs << "</Grid>" << std::endl;
  fs << "</Domain>" << std::endl;
  fs << "</Xdmf>" << std::endl;
}

void
DataExporter::writeXMF(int step, double time) {
  if (!xmf.is_open()) {
    xmf.open(outputDirectory + "data.xmf");
    writeXMFHead(xmf);
    writeXMFStep(xmf, step, time);
    writeXMFTail(xmf);
  } else {
    long pos = xmf.tellp();
    xmf.seekp(pos - 26);
    writeXMFStep(xmf, step, time);
    writeXMFTail(xmf);
  }
}

// Explicit instantiation of templates
template void DataExporter::AddField<Scalar>(
    const std::string &name, const ScalarField<Scalar> &array);
template void DataExporter::AddField<Scalar>(
    const std::string &name, const VectorField<Scalar> &array);
}  // namespace Aperture
