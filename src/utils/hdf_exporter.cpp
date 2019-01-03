// #include "cuda/cudaUtility.h"
#include "utils/hdf_exporter.h"
#include "data/grid_log_sph.h"
#include "data/photons.h"
#include "fmt/core.h"
#include "utils/logger.h"
#include "utils/type_name.h"
// #include "config_file.h"
#include "commandline_args.h"
#include "nlohmann/json.hpp"
#include "cu_sim_data.h"
#include "sim_environment_dev.h"
#include "sim_params.h"
// #include <H5Cpp.h>
#define BOOST_NO_CXX11_SCOPED_ENUMS
#include <boost/filesystem.hpp>
#undef BOOST_NO_CXX11_SCOPED_ENUMS
#include <fstream>

#define H5_USE_BOOST

#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>
#include <time.h>

#include "visit_struct/visit_struct.hpp"

using json = nlohmann::json;
using namespace HighFive;

namespace Aperture {

bool
copyDir(boost::filesystem::path const &source,
        boost::filesystem::path const &destination) {
  namespace fs = boost::filesystem;
  try {
    // Check whether the function call is valid
    if (!fs::exists(source) || !fs::is_directory(source)) {
      std::cerr << "Source directory " << source.string()
                << " does not exist or is not a directory." << '\n';
      return false;
    }
    // if (fs::exists(destination)) {
    //   std::cerr << "Destination directory " << destination.string()
    //             << " already exists." << '\n';
    //   return false;
    // }
    // Create the destination directory
    if (!fs::exists(destination)) {
      if (!fs::create_directory(destination)) {
        std::cerr << "Unable to create destination directory"
                  << destination.string() << '\n';
        return false;
      }
    }
  } catch (fs::filesystem_error const &e) {
    std::cerr << e.what() << '\n';
    return false;
  }
  // Iterate through the source directory
  for (fs::directory_iterator file(source);
       file != fs::directory_iterator(); ++file) {
    try {
      fs::path current(file->path());
      if (fs::is_directory(current)) {
        // Found directory: Recursion
        if (!copyDir(current, destination / current.filename())) {
          return false;
        }
      } else {
        // Found file: Copy
        fs::copy_file(
            current, destination / current.filename(),
            boost::filesystem::copy_option::overwrite_if_exists);
      }
    } catch (fs::filesystem_error const &e) {
      std::cerr << e.what() << '\n';
    }
  }
  return true;
}
// hdf_exporter::hdf_exporter() {}

// hdf_exporter::hdf_exporter(const SimParams &params,
//                            const std::string &dir,
//                            const std::string &prefix, int downsample)
hdf_exporter::hdf_exporter(SimParams &params,
                           uint32_t &timestep)
    : outputDirectory(params.data_dir),
      filePrefix("data"),
      m_params(params),
      downsample_factor(params.downsample) {
  auto &dir = params.data_dir;
  auto downsample = params.downsample;

  // Create the root output directory
  boost::filesystem::path rootPath(dir.c_str());
  boost::system::error_code returnedError;

  boost::filesystem::create_directories(rootPath, returnedError);
  if (outputDirectory.back() != '/') outputDirectory.push_back('/');

  // Setup the grid
  if (params.coord_system == "Cartesian") {
    // grid = std::make_shared<Grid>();
    grid.reset(new Grid());
  } else if (params.coord_system == "LogSpherical") {
    // grid = std::make_shared<Grid_LogSph>();
    grid.reset(new Grid_LogSph());
  } else {
    // grid = std::make_shared<Grid>();
    grid.reset(new Grid());
  }

  grid->init(params);
  for (int i = 0; i < grid->mesh().dim(); i++) {
    grid->mesh().dims[i] =
        params.N[i] / downsample + 2 * params.guard[i];
    grid->mesh().delta[i] *= downsample;
    grid->mesh().inv_delta[i] /= downsample;
  }
}

hdf_exporter::~hdf_exporter() {}

void
hdf_exporter::add_field_output(const std::string &name,
                               const std::string &type,
                               int num_components, field_base *field,
                               int dim, bool sync) {
  auto &mesh = field->grid().mesh();
  if (dim == 3) {
    fieldoutput<3> tempData;
    tempData.name = name;
    tempData.type = type;
    tempData.field = field;
    tempData.sync = sync;
    tempData.f.resize(num_components);
    for (int i = 0; i < num_components; i++) {
      tempData.f[i].resize(
          boost::extents[mesh.dims[2]][mesh.dims[1]][mesh.dims[0]]);
    }
    dbFields3d.push_back(std::move(tempData));
  } else if (dim == 2) {
    fieldoutput<2> tempData;
    tempData.name = name;
    tempData.type = type;
    tempData.field = field;
    tempData.sync = sync;
    tempData.f.resize(num_components);
    for (int i = 0; i < num_components; i++) {
      tempData.f[i].resize(boost::extents[mesh.dims[1]][mesh.dims[0]]);
    }
    dbFields2d.push_back(std::move(tempData));
  } else if (dim == 1) {
    fieldoutput<1> tempData;
    tempData.name = name;
    tempData.type = type;
    tempData.field = field;
    tempData.sync = sync;
    tempData.f.resize(num_components);
  }
  //   if (grid->dim() == 3) {
  //     fieldoutput<3> tempData;
  //     tempData.name = name;
  //     tempData.type = TypeName<T>::Get();
  //     tempData.field = &field;
  //     tempData.f.resize(1);
  //     tempData.f[0].resize(
  //         boost::extents[mesh.dims[2]][mesh.dims[1]][mesh.dims[0]]);
  //     tempData.sync = sync;
  //     dbFields3d.push_back(std::move(tempData));
  //   } else if (grid->dim() == 2) {
  //     fieldoutput<2> tempData;
  //     tempData.name = name;
  //     tempData.type = TypeName<T>::Get();
  //     tempData.field = &field;
  //     tempData.f.resize(1);
  //     tempData.f[0].resize(boost::extents[mesh.dims[1]][mesh.dims[0]]);
  //     tempData.sync = sync;
  //     dbFields2d.push_back(std::move(tempData));
  //   }
  // }
}

void
hdf_exporter::createDirectories() {
  boost::filesystem::path subPath(outputDirectory);
  boost::filesystem::path logPath(outputDirectory + "log/");

  boost::system::error_code returnedError;
  boost::filesystem::create_directories(subPath, returnedError);
  boost::filesystem::create_directories(logPath, returnedError);
}

void
hdf_exporter::copyConfigFile() {
  std::string path = outputDirectory + "config.toml";
  Logger::print_info("{}", path);
  boost::filesystem::copy_file(
      m_params.conf_file, path,
      boost::filesystem::copy_option::overwrite_if_exists);

  boost::filesystem::path submit_path("./submit.cmd");
  if (boost::filesystem::exists(submit_path)) {
    Logger::print_info("Copying submit.cmd");
    boost::filesystem::copy_file(
        "./submit.cmd", outputDirectory + "submit.cmd",
        boost::filesystem::copy_option::overwrite_if_exists);
  }
}

void
hdf_exporter::copySrc() {
  boost::filesystem::path src_path("./src");
  boost::filesystem::path dest_path(outputDirectory + "src");

  copyDir(src_path, dest_path);
}

bool
hdf_exporter::checkDirectories() {
  boost::filesystem::path subPath(outputDirectory);

  return boost::filesystem::exists(outputDirectory);
}

void
hdf_exporter::WriteGrid() {
  if (grid->dim() == 3) {
    auto &mesh = grid->mesh();
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
  } else if (grid->dim() == 2) {
    auto &mesh = grid->mesh();
    boost::multi_array<float, 2> x1_array(
        boost::extents[mesh.dims[1]][mesh.dims[0]]);
    boost::multi_array<float, 2> x2_array(
        boost::extents[mesh.dims[1]][mesh.dims[0]]);

    for (int j = 0; j < mesh.dims[1]; j++) {
      for (int i = 0; i < mesh.dims[0]; i++) {
        if (m_params.coord_system == "LogSpherical") {
          float r = std::exp(mesh.pos(0, i, false));
          float theta = mesh.pos(1, j, false);
          x1_array[j][i] = r * std::sin(theta);
          x2_array[j][i] = r * std::cos(theta);
        } else {
          x1_array[j][i] = mesh.pos(0, i, false);
          x2_array[j][i] = mesh.pos(1, j, false);
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
  }
}

void
hdf_exporter::WriteOutput(int timestep, double time) {
  if (!checkDirectories()) createDirectories();
  // InterpolateFieldValues();
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
  for (auto &f : dbFields2d) {
    int components = f.f.size();
    // if (f.type == "float")
    //   InterpolateFieldValues(f, components, float{});
    // else if (f.type == "double")
    //   InterpolateFieldValues(f, components, double{});
    if (components == 1) {
      DataSet data = datafile.createDataSet<float>(
          f.name, DataSpace::From(f.f[0]));
      data.write(f.f[0]);
    } else {
      for (int n = 0; n < components; n++) {
        DataSet data = datafile.createDataSet<float>(
            f.name + std::to_string(n + 1), DataSpace::From(f.f[n]));
        data.write(f.f[n]);
      }
    }
  }
}

void
hdf_exporter::writeConfig(const SimParams &params) {
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
hdf_exporter::writeXMFHead(std::ofstream &fs) {
  fs << "<?xml version=\"1.0\" ?>" << std::endl;
  fs << "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>" << std::endl;
  fs << "<Xdmf>" << std::endl;
  fs << "<Domain>" << std::endl;
  fs << "<Grid Name=\"Aperture\" GridType=\"Collection\" "
        "CollectionType=\"Temporal\" >"
     << std::endl;
}

void
hdf_exporter::writeXMFStep(std::ofstream &fs, int step, double time) {
  std::string dim_str;
  auto &mesh = grid->mesh();
  if (grid->dim() == 3) {
    dim_str = fmt::format("{} {} {}", mesh.dims[2], mesh.dims[1],
                          mesh.dims[0]);
  } else if (grid->dim() == 2) {
    dim_str = fmt::format("{} {}", mesh.dims[1], mesh.dims[0]);
  }

  fs << "<Grid Name=\"quadmesh\" Type=\"Uniform\">" << std::endl;
  fs << "  <Time Type=\"Single\" Value=\"" << time << "\"/>"
     << std::endl;
  if (grid->dim() == 3) {
    fs << "  <Topology Type=\"3DSMesh\" NumberOfElements=\"" << dim_str
       << "\"/>" << std::endl;
    fs << "  <Geometry GeometryType=\"X_Y_Z\">" << std::endl;
  } else if (grid->dim() == 2) {
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
  if (grid->dim() == 3) {
    fs << "    <DataItem Dimensions=\"" << dim_str
       << "\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">"
       << std::endl;
    fs << "      mesh.h5:x3" << std::endl;
    fs << "    </DataItem>" << std::endl;
  }

  fs << "  </Geometry>" << std::endl;

  for (auto &f : dbFields2d) {
    if (f.f.size() == 1) {
      fs << "  <Attribute Name=\"" << f.name
         << "\" Center=\"Node\" AttributeType=\"Scalar\">" << std::endl;
      fs << "    <DataItem Dimensions=\"" << dim_str
         << "\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">"
         << std::endl;
      fs << fmt::format("      {}{:06d}.h5:{}", filePrefix, step,
                        f.name)
         << std::endl;
      fs << "    </DataItem>" << std::endl;
      fs << "  </Attribute>" << std::endl;
    } else if (f.f.size() == 3) {
      for (int i = 0; i < 3; i++) {
        fs << "  <Attribute Name=\"" << f.name << i + 1
           << "\" Center=\"Node\" AttributeType=\"Scalar\">"
           << std::endl;
        fs << "    <DataItem Dimensions=\"" << dim_str
           << "\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">"
           << std::endl;
        fs << fmt::format("      {}{:06d}.h5:{}{}", filePrefix, step,
                          f.name, i + 1)
           << std::endl;
        fs << "    </DataItem>" << std::endl;
        fs << "  </Attribute>" << std::endl;
      }
    }
  }

  // for (auto &sf : dbScalars2d) {
  //   fs << "  <Attribute Name=\"" << sf.name
  //      << "\" Center=\"Node\" AttributeType=\"Scalar\">" <<
  //      std::endl;
  //   fs << "    <DataItem Dimensions=\"" << dim_str
  //      << "\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">"
  //      << std::endl;
  //   fs << fmt::format("      {}{:06d}.h5:{}", filePrefix, step,
  //   sf.name)
  //      << std::endl;
  //   fs << "    </DataItem>" << std::endl;
  //   fs << "  </Attribute>" << std::endl;
  // }
  // for (auto &sf : dbScalars3d) {
  //   fs << "  <Attribute Name=\"" << sf.name
  //      << "\" Center=\"Node\" AttributeType=\"Scalar\">" <<
  //      std::endl;
  //   fs << "    <DataItem Dimensions=\"" << dim_str
  //      << "\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">"
  //      << std::endl;
  //   fs << fmt::format("      {}{:06d}.h5:{}", filePrefix, step,
  //   sf.name)
  //      << std::endl;
  //   fs << "    </DataItem>" << std::endl;
  //   fs << "  </Attribute>" << std::endl;
  // }
  // for (auto &vf : dbVectors2d) {
  //   fs << "  <Attribute Name=\"" << vf.name
  //      << "1\" Center=\"Node\" AttributeType=\"Scalar\">" <<
  //      std::endl;
  //   fs << "    <DataItem Dimensions=\"" << dim_str
  //      << "\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">"
  //      << std::endl;
  //   fs << fmt::format("      {}{:06d}.h5:{}1", filePrefix, step,
  //                     vf.name)
  //      << std::endl;
  //   fs << "    </DataItem>" << std::endl;
  //   fs << "  </Attribute>" << std::endl;
  //   fs << "  <Attribute Name=\"" << vf.name
  //      << "2\" Center=\"Node\" AttributeType=\"Scalar\">" <<
  //      std::endl;
  //   fs << "    <DataItem Dimensions=\"" << dim_str
  //      << "\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">"
  //      << std::endl;
  //   fs << fmt::format("      {}{:06d}.h5:{}2", filePrefix, step,
  //                     vf.name)
  //      << std::endl;
  //   fs << "    </DataItem>" << std::endl;
  //   fs << "  </Attribute>" << std::endl;
  //   fs << "  <Attribute Name=\"" << vf.name
  //      << "3\" Center=\"Node\" AttributeType=\"Scalar\">" <<
  //      std::endl;
  //   fs << "    <DataItem Dimensions=\"" << dim_str
  //      << "\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">"
  //      << std::endl;
  //   fs << fmt::format("      {}{:06d}.h5:{}3", filePrefix, step,
  //                     vf.name)
  //      << std::endl;
  //   fs << "    </DataItem>" << std::endl;
  //   fs << "  </Attribute>" << std::endl;
  // }

  // for (auto &vf : dbVectors3d) {
  //   fs << "  <Attribute Name=\"" << vf.name
  //      << "1\" Center=\"Node\" AttributeType=\"Scalar\">" <<
  //      std::endl;
  //   fs << "    <DataItem Dimensions=\"" << dim_str
  //      << "\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">"
  //      << std::endl;
  //   fs << fmt::format("      {}{:06d}.h5:{}1", filePrefix, step,
  //                     vf.name)
  //      << std::endl;
  //   fs << "    </DataItem>" << std::endl;
  //   fs << "  </Attribute>" << std::endl;
  //   fs << "  <Attribute Name=\"" << vf.name
  //      << "2\" Center=\"Node\" AttributeType=\"Scalar\">" <<
  //      std::endl;
  //   fs << "    <DataItem Dimensions=\"" << dim_str
  //      << "\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">"
  //      << std::endl;
  //   fs << fmt::format("      {}{:06d}.h5:{}2", filePrefix, step,
  //                     vf.name)
  //      << std::endl;
  //   fs << "    </DataItem>" << std::endl;
  //   fs << "  </Attribute>" << std::endl;
  //   fs << "  <Attribute Name=\"" << vf.name
  //      << "3\" Center=\"Node\" AttributeType=\"Scalar\">" <<
  //      std::endl;
  //   fs << "    <DataItem Dimensions=\"" << dim_str
  //      << "\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">"
  //      << std::endl;
  //   fs << fmt::format("      {}{:06d}.h5:{}3", filePrefix, step,
  //                     vf.name)
  //      << std::endl;
  //   fs << "    </DataItem>" << std::endl;
  //   fs << "  </Attribute>" << std::endl;
  // }

  fs << "</Grid>" << std::endl;
}

void
hdf_exporter::writeXMFTail(std::ofstream &fs) {
  fs << "</Grid>" << std::endl;
  fs << "</Domain>" << std::endl;
  fs << "</Xdmf>" << std::endl;
}

void
hdf_exporter::writeXMF(uint32_t step, double time) {
  if (!xmf.is_open()) {
    xmf.open(outputDirectory + "data.xmf");
  }
  if (step == 0) {
    writeXMFHead(xmf);
    writeXMFStep(xmf, step, time);
    writeXMFTail(xmf);
  } else {
    // long pos = xmf.tellp();
    xmf.seekp(-26, std::ios_base::end);
    writeXMFStep(xmf, step, time);
    writeXMFTail(xmf);
  }
}

void
hdf_exporter::writeSnapshot(Environment &env, cu_sim_data &data,
                            uint32_t timestep) {
  File snapshotfile(
      // fmt::format("{}snapshot{:06d}.h5", outputDirectory, timestep)
      fmt::format("{}snapshot.h5", outputDirectory).c_str(),
      File::ReadWrite | File::Create | File::Truncate);

  // Write background fields from environment
  size_t grid_size = data.E.grid().size();
  data.Ebg.sync_to_host();
  DataSet data_bg_E1 =
      snapshotfile.createDataSet<Scalar>("bg_E1", DataSpace(grid_size));
  data_bg_E1.write(data.Ebg.data(0).data());
  DataSet data_bg_E2 =
      snapshotfile.createDataSet<Scalar>("bg_E2", DataSpace(grid_size));
  data_bg_E2.write(data.Ebg.data(1).data());
  DataSet data_bg_E3 =
      snapshotfile.createDataSet<Scalar>("bg_E3", DataSpace(grid_size));
  data_bg_E3.write(data.Ebg.data(2).data());
  data.Bbg.sync_to_host();
  DataSet data_bg_B1 =
      snapshotfile.createDataSet<Scalar>("bg_B1", DataSpace(grid_size));
  data_bg_B1.write(data.Bbg.data(0).data());
  DataSet data_bg_B2 =
      snapshotfile.createDataSet<Scalar>("bg_B2", DataSpace(grid_size));
  data_bg_B2.write(data.Bbg.data(1).data());
  DataSet data_bg_B3 =
      snapshotfile.createDataSet<Scalar>("bg_B3", DataSpace(grid_size));
  data_bg_B3.write(data.Bbg.data(2).data());

  // Write sim data
  // Write field values
  data.E.sync_to_host();
  DataSet data_E1 =
      snapshotfile.createDataSet<Scalar>("E1", DataSpace(grid_size));
  data_E1.write(data.E.data(0).data());
  DataSet data_E2 =
      snapshotfile.createDataSet<Scalar>("E2", DataSpace(grid_size));
  data_E2.write(data.E.data(1).data());
  DataSet data_E3 =
      snapshotfile.createDataSet<Scalar>("E3", DataSpace(grid_size));
  data_E3.write(data.E.data(2).data());
  data.B.sync_to_host();
  DataSet data_B1 =
      snapshotfile.createDataSet<Scalar>("B1", DataSpace(grid_size));
  data_B1.write(data.B.data(0).data());
  DataSet data_B2 =
      snapshotfile.createDataSet<Scalar>("B2", DataSpace(grid_size));
  data_B2.write(data.B.data(1).data());
  DataSet data_B3 =
      snapshotfile.createDataSet<Scalar>("B3", DataSpace(grid_size));
  data_B3.write(data.B.data(2).data());
  data.J.sync_to_host();
  DataSet data_J1 =
      snapshotfile.createDataSet<Scalar>("J1", DataSpace(grid_size));
  data_J1.write(data.J.data(0).data());
  DataSet data_J2 =
      snapshotfile.createDataSet<Scalar>("J2", DataSpace(grid_size));
  data_J2.write(data.J.data(1).data());
  DataSet data_J3 =
      snapshotfile.createDataSet<Scalar>("J3", DataSpace(grid_size));
  data_J3.write(data.J.data(2).data());

  for (int i = 0; i < data.num_species; i++) {
    data.Rho[i].sync_to_host();
    DataSet data_Rho = snapshotfile.createDataSet<Scalar>(
        fmt::format("Rho{}", i), DataSpace(grid_size));
    data_Rho.write(data.Rho[i].data().data());
  }
  DataSet data_devId = snapshotfile.createDataSet<int>(
      "devId", DataSpace::From(data.devId));
  data_devId.write(data.devId);

  // Write particle data
  size_t ptcNum = data.particles.number();
  DataSet data_ptcNum = snapshotfile.createDataSet<size_t>(
      "ptcNum", DataSpace::From(ptcNum));
  data_ptcNum.write(ptcNum);
  Logger::print_info("Writing {} particles to snapshot", ptcNum);

  size_t phNum = data.photons.number();
  DataSet data_phNum = snapshotfile.createDataSet<size_t>(
      "phNum", DataSpace::From(phNum));
  data_phNum.write(phNum);
  Logger::print_info("Writing {} photons to snapshot", phNum);

  std::vector<double> buffer(std::max(ptcNum, phNum));
  visit_struct::for_each(
      data.particles.data(),
      [&snapshotfile, &buffer, &ptcNum](const char *name, auto &x) {
        typedef
            typename std::remove_reference<decltype(*x)>::type x_type;
        DataSet ptc_data = snapshotfile.createDataSet<x_type>(
            fmt::format("ptc_{}", name), DataSpace(ptcNum));
        cudaMemcpy(buffer.data(), x, ptcNum * sizeof(x_type),
                   cudaMemcpyDeviceToHost);
        ptc_data.write(reinterpret_cast<x_type *>(buffer.data()));
      });
  visit_struct::for_each(
      data.photons.data(),
      [&snapshotfile, &buffer, &phNum](const char *name, auto &x) {
        typedef
            typename std::remove_reference<decltype(*x)>::type x_type;
        DataSet ph_data = snapshotfile.createDataSet<x_type>(
            fmt::format("ph_{}", name), DataSpace(phNum));
        cudaMemcpy(buffer.data(), x, phNum * sizeof(x_type),
                   cudaMemcpyDeviceToHost);
        ph_data.write(reinterpret_cast<x_type *>(buffer.data()));
      });

  // Write current simulation timestep and other info
  DataSet data_timestep = snapshotfile.createDataSet<uint32_t>(
      "timestep", DataSpace::From(timestep));
  data_timestep.write(timestep);
}

void
hdf_exporter::load_from_snapshot(Environment &env, cu_sim_data &data,
                                 uint32_t &timestep) {
  File snapshotfile(
      // fmt::format("{}snapshot{:06d}.h5", outputDirectory, timestep)
      fmt::format("{}snapshot.h5", outputDirectory).c_str(),
      File::ReadOnly);

  size_t grid_size = data.E.grid().size();
  size_t ptcNum, phNum;
  int devId;

  // Read the scalars first
  DataSet data_timestep = snapshotfile.getDataSet("timestep");
  data_timestep.read(timestep);
  DataSet data_ptcNum = snapshotfile.getDataSet("ptcNum");
  data_ptcNum.read(ptcNum);
  DataSet data_phNum = snapshotfile.getDataSet("phNum");
  data_phNum.read(phNum);
  DataSet data_devId = snapshotfile.getDataSet("devId");
  data_devId.read(devId);

  // Read particle data
  std::vector<double> buffer(std::max(ptcNum, phNum));
  data.particles.set_num(ptcNum);
  data.photons.set_num(phNum);

  visit_struct::for_each(
      data.particles.data(),
      [&snapshotfile, &buffer, &ptcNum](const char *name, auto &x) {
        typedef
            typename std::remove_reference<decltype(*x)>::type x_type;
        DataSet ptc_data =
            snapshotfile.getDataSet(fmt::format("ptc_{}", name));
        ptc_data.read(reinterpret_cast<x_type *>(buffer.data()));
        cudaMemcpy(x, buffer.data(), ptcNum * sizeof(x_type),
                   cudaMemcpyHostToDevice);
      });
  visit_struct::for_each(
      data.photons.data(),
      [&snapshotfile, &buffer, &phNum](const char *name, auto &x) {
        typedef
            typename std::remove_reference<decltype(*x)>::type x_type;
        DataSet ph_data =
            snapshotfile.getDataSet(fmt::format("ph_{}", name));
        ph_data.read(reinterpret_cast<x_type *>(buffer.data()));
        cudaMemcpy(x, buffer.data(), phNum * sizeof(x_type),
                   cudaMemcpyHostToDevice);
      });

  // Read field data
  DataSet data_bg_B1 = snapshotfile.getDataSet("bg_B1");
  data_bg_B1.read(data.Bbg.data(0).data());
  DataSet data_bg_B2 = snapshotfile.getDataSet("bg_B2");
  data_bg_B2.read(data.Bbg.data(1).data());
  DataSet data_bg_B3 = snapshotfile.getDataSet("bg_B3");
  data_bg_B3.read(data.Bbg.data(2).data());
  DataSet data_bg_E1 = snapshotfile.getDataSet("bg_E1");
  data_bg_E1.read(data.Ebg.data(0).data());
  DataSet data_bg_E2 = snapshotfile.getDataSet("bg_E2");
  data_bg_E2.read(data.Ebg.data(1).data());
  DataSet data_bg_E3 = snapshotfile.getDataSet("bg_E3");
  data_bg_E3.read(data.Ebg.data(2).data());

  data.Bbg.sync_to_device();
  data.Ebg.sync_to_device();

  DataSet data_B1 = snapshotfile.getDataSet("B1");
  data_B1.read(data.B.data(0).data());
  DataSet data_B2 = snapshotfile.getDataSet("B2");
  data_B2.read(data.B.data(1).data());
  DataSet data_B3 = snapshotfile.getDataSet("B3");
  data_B3.read(data.B.data(2).data());
  DataSet data_E1 = snapshotfile.getDataSet("E1");
  data_E1.read(data.E.data(0).data());
  DataSet data_E2 = snapshotfile.getDataSet("E2");
  data_E2.read(data.E.data(1).data());
  DataSet data_E3 = snapshotfile.getDataSet("E3");
  data_E3.read(data.E.data(2).data());
  DataSet data_J1 = snapshotfile.getDataSet("J1");
  data_J1.read(data.J.data(0).data());
  DataSet data_J2 = snapshotfile.getDataSet("J2");
  data_J2.read(data.J.data(1).data());
  DataSet data_J3 = snapshotfile.getDataSet("J3");
  data_J3.read(data.J.data(2).data());
  data.B.sync_to_device();
  data.E.sync_to_device();
  data.J.sync_to_device();

  for (int i = 0; i < data.num_species; i++) {
    DataSet data_rho = snapshotfile.getDataSet(fmt::format("Rho{}", i));
    data_rho.read(data.Rho[i].data().data());
    data.Rho[i].sync_to_device();
  }
}

void
hdf_exporter::prepareXMFrestart(uint32_t restart_step,
                                int data_interval) {
  boost::filesystem::path xmf_file(outputDirectory + "data.xmf");
  boost::filesystem::path xmf_bak(outputDirectory + "data.xmf.bak");
  boost::filesystem::remove(xmf_bak);
  boost::filesystem::rename(xmf_file, xmf_bak);

  std::ifstream xmf_in;
  xmf_in.open(xmf_bak.c_str());

  xmf.open(xmf_file.c_str());

  int n = -1;
  int num_outputs = restart_step / data_interval;
  std::string line;
  bool in_step = false;
  while (std::getline(xmf_in, line)) {
    if (line == "<Grid Name=\"quadmesh\" Type=\"Uniform\">") {
      n += 1;
      in_step = true;
    }
    xmf << line << std::endl;
    if (line == "</Grid>") in_step = false;
    if (n >= num_outputs && !in_step) break;
  }
  writeXMFTail(xmf);

  xmf_in.close();
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
// DataExporter::AddArray(const std::string &name, cu_multi_array<T>
// &array)
// {
//   int ndims = array.dim();
//   int *dims = new int[ndims];
//   for (int i = 0; i < ndims; i++) dims[i] = array.extent()[i];

//   AddArray(name, array.data(), dims, ndims);

//   delete[] dims;
// }

// template <typename T>
// void
// DataExporter::AddArray(const std::string &name, cu_vector_field<T>
// &field,
//                        int component) {
//   AddArray(name, field.data(component));
// }

// template <typename T>
// void hdf_exporter::AddField(const std::string &name, cu_scalar_field<T>
// &field,
//                             bool sync) {
//   auto &mesh = grid->mesh();

//   if (grid->dim() == 3) {
//     // sfieldoutput3d<T> tempData;
//     // tempData.name = name;
//     // tempData.field = &field;

//     // tempData.f.resize(
//     //     boost::extents[mesh.dims[2]][mesh.dims[1]][mesh.dims[0]]);
//     // tempData.sync = sync;
//     // if (std::is_same<T, double>::value)
//     //   dbScalars3d.push_back(std::move(tempData));
//     // else
//     //   dbScalars3f.push_back(std::move(tempData));
//     fieldoutput<3> tempData;
//     tempData.name = name;
//     tempData.type = TypeName<T>::Get();
//     tempData.field = &field;
//     tempData.f.resize(1);
//     tempData.f[0].resize(
//         boost::extents[mesh.dims[2]][mesh.dims[1]][mesh.dims[0]]);
//     tempData.sync = sync;
//     dbFields3d.push_back(std::move(tempData));
//   } else if (grid->dim() == 2) {
//     fieldoutput<2> tempData;
//     tempData.name = name;
//     tempData.type = TypeName<T>::Get();
//     tempData.field = &field;
//     tempData.f.resize(1);
//     tempData.f[0].resize(boost::extents[mesh.dims[1]][mesh.dims[0]]);
//     tempData.sync = sync;
//     dbFields2d.push_back(std::move(tempData));
//   }
// }

// template <typename T>
// void hdf_exporter::AddField(const std::string &name, cu_vector_field<T>
// &field,
//                             bool sync) {
//   auto &mesh = grid->mesh();

//   if (grid->dim() == 3) {
//     fieldoutput<3> tempData;
//     tempData.name = name;
//     tempData.type = TypeName<T>::Get();
//     tempData.field = &field;
//     tempData.f.resize(3);
//     for (int i = 0; i < 3; i++)
//       tempData.f[i].resize(
//           boost::extents[mesh.dims[2]][mesh.dims[1]][mesh.dims[0]]);
//     tempData.sync = sync;
//     dbFields3d.push_back(std::move(tempData));
//   } else if (grid->dim() == 2) {
//     fieldoutput<2> tempData;
//     tempData.name = name;
//     tempData.type = TypeName<T>::Get();
//     tempData.field = &field;
//     tempData.f.resize(3);
//     for (int i = 0; i < 3; i++)
//       tempData.f[i].resize(boost::extents[mesh.dims[1]][mesh.dims[0]]);
//     tempData.sync = sync;
//     dbFields2d.push_back(std::move(tempData));
//   }
// }

// void
// DataExporter::InterpolateFieldValues() {
//   for (auto &sf : dbScalars3d) {
//     if (sf.type == 8)
//       cu_scalar_field<double> *fptr = (cu_scalar_field<double> *)sf.field;
//     else
//       cu_scalar_field<float> *fptr = (cu_scalar_field<float> *)sf.field;
//     if (sf.sync) fptr->sync_to_host();
//     auto &mesh = fptr->grid().mesh();
//     for (int k = 0; k < mesh.reduced_dim(2); k += downsample_factor)
//     {
//       for (int j = 0; j < mesh.reduced_dim(1); j +=
//       downsample_factor) {
//         for (int i = 0; i < mesh.reduced_dim(0);
//              i += downsample_factor) {
//           sf.f[k / downsample_factor + mesh.guard[2]]
//               [j / downsample_factor + mesh.guard[1]]
//               [i / downsample_factor + mesh.guard[0]] = (*fptr)(
//               i + mesh.guard[0], j + mesh.guard[1], k +
//               mesh.guard[2]);
//         }
//       }
//     }
//   }

//   for (auto &vf : dbVectors3d) {
//     if (vf.type == 8)
//       cu_vector_field<double> *fptr = (cu_vector_field<double> *)vf.field;
//     else
//       cu_vector_field<float> *fptr = (cu_vector_field<float> *)vf.field;
//     if (vf.sync) vf.field->sync_to_host();
//     auto &mesh = vf.field->grid().mesh();
//     for (int k = 0; k < mesh.reduced_dim(2); k += downsample_factor)
//     {
//       for (int j = 0; j < mesh.reduced_dim(1); j +=
//       downsample_factor) {
//         for (int i = 0; i < mesh.reduced_dim(0);
//              i += downsample_factor) {
//           vf.f1[k / downsample_factor + mesh.guard[2]]
//                [j / downsample_factor + mesh.guard[1]]
//                [i / downsample_factor + mesh.guard[0]] =
//               (*vf.field)(0, i + mesh.guard[0], j + mesh.guard[1],
//                           k + mesh.guard[2]);
//           vf.f2[k / downsample_factor + mesh.guard[2]]
//                [j / downsample_factor + mesh.guard[1]]
//                [i / downsample_factor + mesh.guard[0]] =
//               (*vf.field)(1, i + mesh.guard[0], j + mesh.guard[1],
//                           k + mesh.guard[2]);
//           vf.f3[k / downsample_factor + mesh.guard[2]]
//                [j / downsample_factor + mesh.guard[1]]
//                [i / downsample_factor + mesh.guard[0]] =
//               (*vf.field)(2, i + mesh.guard[0], j + mesh.guard[1],
//                           k + mesh.guard[2]);
//         }
//       }
//     }
//   }

//   for (auto &sf : dbScalars2d) {
//     if (sf.sync) sf.field->sync_to_host();
//     auto &mesh = sf.field->grid().mesh();
//     for (int j = 0; j < mesh.reduced_dim(1); j += downsample_factor)
//     {
//       for (int i = 0; i < mesh.reduced_dim(0); i +=
//       downsample_factor) {
//         sf.f[j / downsample_factor + mesh.guard[1]]
//             [i / downsample_factor + mesh.guard[0]] =
//             (*sf.field)(i + mesh.guard[0], j + mesh.guard[1]);
//       }
//       // for (int i = 0; i < mesh.reduced_dim(0); i +=
//       // downsample_factor) {
//       //   sf.f[mesh.guard[1] - 1]
//       //       [i / downsample_factor + mesh.guard[0]] =
//       //       (*sf.field)(i + mesh.guard[0], mesh.guard[1] - 1);
//       // }
//     }
//   }

//   for (auto &vf : dbVectors2d) {
//     if (vf.sync) vf.field->sync_to_host();
//     // Logger::print_info("Writing {}", vf.name);
//     auto &mesh = vf.field->grid().mesh();
//     for (int j = 0; j < mesh.reduced_dim(1); j += downsample_factor)
//     {
//       for (int i = 0; i < mesh.reduced_dim(0); i +=
//       downsample_factor) {
//         vf.f1[j / downsample_factor + mesh.guard[1]]
//              [i / downsample_factor + mesh.guard[0]] =
//             (*vf.field)(0, i + mesh.guard[0], j + mesh.guard[1]);
//         // std::cout << vf.f1[j / downsample_factor + mesh.guard[1]]
//         //     [i / downsample_factor + mesh.guard[0]] << std::endl;
//         vf.f2[j / downsample_factor + mesh.guard[1]]
//              [i / downsample_factor + mesh.guard[0]] =
//             (*vf.field)(1, i + mesh.guard[0], j + mesh.guard[1]);
//         vf.f3[j / downsample_factor + mesh.guard[1]]
//              [i / downsample_factor + mesh.guard[0]] =
//             (*vf.field)(2, i + mesh.guard[0], j + mesh.guard[1]);
//       }
//       // for (int i = 0; i < mesh.reduced_dim(0); i +=
//       // downsample_factor) {
//       //   vf.f1[mesh.guard[1] - 1]
//       //       [i / downsample_factor + mesh.guard[0]] =
//       //       (*vf.field)(0, i + mesh.guard[0], mesh.guard[1] - 1);
//       //   vf.f2[mesh.guard[1] - 1]
//       //       [i / downsample_factor + mesh.guard[0]] =
//       //       (*vf.field)(1, i + mesh.guard[0], mesh.guard[1] - 1);
//       //   vf.f3[mesh.guard[1] - 1]
//       //       [i / downsample_factor + mesh.guard[0]] =
//       //       (*vf.field)(2, i + mesh.guard[0], mesh.guard[1] - 1);
//       // }
//     }
//   }
// }

// template <typename T>
// void hdf_exporter::InterpolateFieldValues(fieldoutput<2> &field, int
// components,
//                                           T t) {
//   if (components == 1) {
//     auto fptr = dynamic_cast<cu_scalar_field<T> *>(field.field);
//     if (field.sync)
//       fptr->sync_to_host();
//     auto &mesh = fptr->grid().mesh();
//     for (int j = 0; j < mesh.reduced_dim(1); j += downsample_factor)
//     {
//       for (int i = 0; i < mesh.reduced_dim(0); i +=
//       downsample_factor) {
//         field.f[0][j / downsample_factor + mesh.guard[1]]
//                [i / downsample_factor + mesh.guard[0]] = 0.0;
//         for (int n2 = 0; n2 < downsample_factor; n2++) {
//           for (int n1 = 0; n1 < downsample_factor; n1++) {
//             field.f[0][j / downsample_factor + mesh.guard[1]]
//                    [i / downsample_factor + mesh.guard[0]] +=
//                 (*fptr)(i + n1 + mesh.guard[0], j + n2 +
//                 mesh.guard[1]) / square(downsample_factor);
//           }
//         }
//       }
//       // for (int i = 0; i < mesh.reduced_dim(0); i +=
//       // downsample_factor) {
//       //   field.f[0][mesh.guard[1] - 1]
//       //          [i / downsample_factor + mesh.guard[0]] =
//       //       (*fptr)(i + mesh.guard[0], mesh.guard[1] - 1);
//       // }
//     }
//   } else if (components == 3) {
//     auto fptr = dynamic_cast<cu_vector_field<T> *>(field.field);
//     if (field.sync)
//       fptr->sync_to_host();
//     auto &mesh = fptr->grid().mesh();
//     for (int j = 0; j < mesh.reduced_dim(1); j += downsample_factor)
//     {
//       for (int i = 0; i < mesh.reduced_dim(0); i +=
//       downsample_factor) {
//         field.f[0][j / downsample_factor + mesh.guard[1]]
//                [i / downsample_factor + mesh.guard[0]] = 0.0;
//         field.f[1][j / downsample_factor + mesh.guard[1]]
//                [i / downsample_factor + mesh.guard[0]] = 0.0;
//         field.f[2][j / downsample_factor + mesh.guard[1]]
//                [i / downsample_factor + mesh.guard[0]] = 0.0;
//         for (int n2 = 0; n2 < downsample_factor; n2++) {
//           for (int n1 = 0; n1 < downsample_factor; n1++) {
//             field.f[0][j / downsample_factor + mesh.guard[1]]
//                    [i / downsample_factor + mesh.guard[0]] +=
//                 (*fptr)(0, i + n1 + mesh.guard[0], j + n2 +
//                 mesh.guard[1]) / square(downsample_factor);
//             // std::cout << vf.f1[j / downsample_factor +
//             mesh.guard[1]]
//             //     [i / downsample_factor + mesh.guard[0]] <<
//             std::endl; field.f[1][j / downsample_factor +
//             mesh.guard[1]]
//                    [i / downsample_factor + mesh.guard[0]] +=
//                 (*fptr)(1, i + n1 + mesh.guard[0], j + n2 +
//                 mesh.guard[1]) / square(downsample_factor);

//             field.f[2][j / downsample_factor + mesh.guard[1]]
//                    [i / downsample_factor + mesh.guard[0]] +=
//                 (*fptr)(2, i + n1 + mesh.guard[0], j + n2 +
//                 mesh.guard[1]) / square(downsample_factor);
//           }
//         }
//       }
//       // for (int i = 0; i < mesh.reduced_dim(0); i +=
//       // downsample_factor) {
//       //   field.f[0][mesh.guard[1] - 1]
//       //          [i / downsample_factor + mesh.guard[0]] =
//       //       (*fptr)(0, i + mesh.guard[0], mesh.guard[1] - 1);
//       //   field.f[1][mesh.guard[1] - 1]
//       //          [i / downsample_factor + mesh.guard[0]] =
//       //       (*fptr)(1, i + mesh.guard[0], mesh.guard[1] - 1);
//       //   field.f[2][mesh.guard[1] - 1]
//       //          [i / downsample_factor + mesh.guard[0]] =
//       //       (*fptr)(2, i + mesh.guard[0], mesh.guard[1] - 1);
//       // }
//     }
//   }
// }

// void hdf_exporter::AddParticleArray(const std::string &name,
//                                     const Particles &ptc) {
//   ptcoutput<Particles> temp;
//   temp.name = name;
//   temp.ptc = &ptc;
//   temp.data_x = std::vector<float>(MAX_TRACKED);
//   temp.data_p = std::vector<float>(MAX_TRACKED);

//   dbPtcData.push_back(std::move(temp));
// }

// void hdf_exporter::AddParticleArray(const std::string &name,
//                                     const Photons &ptc) {
//   ptcoutput<Photons> temp;
//   temp.name = name;
//   temp.ptc = &ptc;
//   temp.data_x = std::vector<float>(MAX_TRACKED);
//   temp.data_p = std::vector<float>(MAX_TRACKED);
//   temp.data_l = std::vector<float>(MAX_TRACKED);

//   dbPhotonData.push_back(std::move(temp));
// }

// Explicit instantiation of templates
// template void hdf_exporter::AddField<double>(const std::string &name,
//                                              cu_scalar_field<double>
//                                              &array, bool sync);
// template void hdf_exporter::AddField<double>(const std::string &name,
//                                              cu_vector_field<double>
//                                              &array, bool sync);
// template void hdf_exporter::AddField<float>(const std::string &name,
//                                             cu_scalar_field<float>
//                                             &array, bool sync);
// template void hdf_exporter::AddField<float>(const std::string &name,
//                                             cu_vector_field<float>
//                                             &array, bool sync);
}  // namespace Aperture
