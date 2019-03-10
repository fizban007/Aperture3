// #include "cuda/cudaUtility.h"
#include "grids/grid_log_sph.h"
#include "utils/hdf_exporter.h"
// #include "data/photons_dev.h"
#include "fmt/core.h"
#include "utils/filesystem.h"
#include "utils/logger.h"
#include "utils/type_name.h"
// #include "config_file.h"
#include "commandline_args.h"
#include "nlohmann/json.hpp"
// #include "cu_sim_data.h"
// #include "sim_environment_dev.h"
#include "sim_params.h"
#include <fstream>
#include <vector>

#define H5_USE_BOOST

#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>
#include <time.h>

#include "visit_struct/visit_struct.hpp"

using json = nlohmann::json;
using namespace HighFive;

namespace Aperture {

// hdf_exporter::hdf_exporter() {}

// hdf_exporter::hdf_exporter(const SimParams &params,
//                            const std::string &dir,
//                            const std::string &prefix, int downsample)
template <typename T>
hdf_exporter<T>::hdf_exporter(SimParams &params, uint32_t &timestep)
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
  orig_grid.reset(new Grid());

  grid->init(params);
  orig_grid->init(params);
  for (int i = 0; i < grid->mesh().dim(); i++) {
    grid->mesh().dims[i] =
        params.N[i] / downsample + 2 * params.guard[i];
    grid->mesh().delta[i] *= downsample;
    grid->mesh().inv_delta[i] /= downsample;
  }
  Logger::print_info("initialize output {}d grid of size {}x{}",
                     grid->mesh().dim(), grid->mesh().dims[1],
                     grid->mesh().dims[0]);
}

template <typename T>
hdf_exporter<T>::~hdf_exporter() {}

template <typename T>
void
hdf_exporter<T>::add_field_output(const std::string &name,
                                  const std::string &type,
                                  int num_components, field_base *field,
                                  int dim, bool sync) {
  auto &mesh = grid->mesh();
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
      // Logger::print_info("mesh sizes are {}x{}, f[i] size is {}",
      // mesh.dims[1], mesh.dims[0], tempData.f[i].size());
    }
    dbFields2d.push_back(std::move(tempData));
  } else if (dim == 1) {
    fieldoutput<1> tempData;
    tempData.name = name;
    tempData.type = type;
    tempData.field = field;
    tempData.sync = sync;
    tempData.f.resize(num_components);
    for (int i = 0; i < num_components; i++) {
      tempData.f[i].resize(boost::extents[mesh.dims[0]]);
    }
    dbFields1d.push_back(std::move(tempData));
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

template <typename T>
void
hdf_exporter<T>::add_ptc_output(const std::string &name,
                                const std::string &type,
                                particle_interface *ptc) {}

template <typename T>
void
hdf_exporter<T>::add_ptc_output_1d(const std::string &name,
                                   const std::string &type,
                                   particle_interface *ptc) {
  ptcoutput_1d tmp_data;
  tmp_data.name = name;
  tmp_data.type = type;
  tmp_data.ptc = ptc;
  tmp_data.x = std::vector<float>(MAX_TRACKED);
  tmp_data.p = std::vector<float>(MAX_TRACKED);

  dbPtcData1d.push_back(tmp_data);
}

template <typename T>
void
hdf_exporter<T>::add_ptc_output_2d(const std::string &name,
                                   const std::string &type,
                                   particle_interface *ptc) {
  ptcoutput_2d tmp_data;
  tmp_data.name = name;
  tmp_data.type = type;
  tmp_data.ptc = ptc;
  tmp_data.x1 = std::vector<float>(MAX_TRACKED);
  tmp_data.x2 = std::vector<float>(MAX_TRACKED);
  tmp_data.x3 = std::vector<float>(MAX_TRACKED);
  tmp_data.p1 = std::vector<float>(MAX_TRACKED);
  tmp_data.p2 = std::vector<float>(MAX_TRACKED);
  tmp_data.p3 = std::vector<float>(MAX_TRACKED);

  dbPtcData2d.push_back(tmp_data);
}

template <typename DerivedClass>
template <typename T>
void
hdf_exporter<DerivedClass>::add_array_output(const std::string &name,
                                             multi_array<T> &array) {
  if (array.dim() == 2) {
    arrayoutput<T, 2> tmp;
    tmp.name = name;
    tmp.array = &array;
    tmp.f.resize(boost::extents[array.extent().y][array.extent().x]);
    dbfloat2d.push_back(std::move(tmp));
  }
}

template <typename T>
void
hdf_exporter<T>::createDirectories() {
  boost::filesystem::path subPath(outputDirectory);
  boost::filesystem::path logPath(outputDirectory + "log/");

  boost::system::error_code returnedError;
  boost::filesystem::create_directories(subPath, returnedError);
  boost::filesystem::create_directories(logPath, returnedError);
}

template <typename T>
void
hdf_exporter<T>::copyConfigFile() {
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

template <typename T>
void
hdf_exporter<T>::copySrc() {
  boost::filesystem::path src_path("./src");
  boost::filesystem::path dest_path(outputDirectory + "src");

  copyDir(src_path, dest_path);
}

template <typename T>
bool
hdf_exporter<T>::checkDirectories() {
  boost::filesystem::path subPath(outputDirectory);

  return boost::filesystem::exists(outputDirectory);
}

template <typename T>
void
hdf_exporter<T>::WriteGrid() {
  auto &mesh = grid->mesh();
  if (grid->dim() == 3) {
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
  } else if (grid->dim() == 1) {
    std::vector<float> x_array(mesh.dims[0]);

    for (int i = 0; i < mesh.dims[0]; i++) {
      x_array[i] = mesh.pos(0, i, false);
    }

    std::string meshfilename = outputDirectory + "mesh.h5";
    Logger::print_info("{}", meshfilename);
    File meshfile(meshfilename.c_str(),
                  File::ReadWrite | File::Create | File::Truncate);
    DataSet mesh_x1 =
        meshfile.createDataSet<float>("x1", DataSpace::From(x_array));
    mesh_x1.write(x_array);
  }
}

template <typename T>
void
hdf_exporter<T>::WriteOutput(int timestep, double time) {
  if (!checkDirectories()) createDirectories();
  // InterpolateFieldValues();
  File datafile(fmt::format("{}{}{:06d}.h5", outputDirectory,
                            filePrefix, timestep)
                    .c_str(),
                File::ReadWrite | File::Create | File::Truncate);

  for (auto &f : dbFields1d) {
    int components = f.f.size();
    T *self = static_cast<T *>(this);
    if (f.type == "float") {
      self->interpolate_field_values(f, components, float{});
    } else if (f.type == "double") {
      self->interpolate_field_values(f, components, double{});
    }
    if (components == 1) {
      DataSet data = datafile.createDataSet<float>(
          f.name, DataSpace::From(f.f[0]));
      data.write(f.f[0]);
    } else {
      for (int n = 0; n < 1; n++) {
        DataSet data = datafile.createDataSet<float>(
            f.name + std::to_string(n + 1), DataSpace::From(f.f[n]));
        data.write(f.f[n]);
      }
    }
  }
  for (auto &f : dbFields2d) {
    int components = f.f.size();
    T *self = static_cast<T *>(this);
    if (f.type == "float") {
      self->interpolate_field_values(f, components, float{});
    } else if (f.type == "double") {
      self->interpolate_field_values(f, components, double{});
    }
    if (components == 1) {
      // Logger::print_info("Creating dataset for {}", f.name);
      DataSet data = datafile.createDataSet<float>(
          f.name, DataSpace::From(f.f[0]));
      data.write(f.f[0]);
    } else {
      // Logger::print_info("Creating dataset for {}", f.name);
      for (int n = 0; n < components; n++) {
        // Logger::print_info("{}, {}", n, f.f[n].size());
        DataSet data = datafile.createDataSet<float>(
            f.name + std::to_string(n + 1), DataSpace::From(f.f[n]));
        // Logger::print_info("dataset created");
        data.write(f.f[n]);
      }
    }
  }
  for (auto &f : dbfloat2d) {
    // copy field values to ouput buffer
    for (int j = 0; j < f.array->extent().y; j++) {
      for (int i = 0; i < f.array->extent().x; i++) {
        f.f[j][i] = (*f.array)(i, j);
      }
    }
    DataSet data = datafile.createDataSet<float>(f.name, DataSpace::From(f.f));
    data.write(f.f);
  }
}

template <typename T>
void
hdf_exporter<T>::writeConfig(const SimParams &params) {
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

template <typename T>
void
hdf_exporter<T>::writeXMFHead(std::ofstream &fs) {
  fs << "<?xml version=\"1.0\" ?>" << std::endl;
  fs << "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>" << std::endl;
  fs << "<Xdmf>" << std::endl;
  fs << "<Domain>" << std::endl;
  fs << "<Grid Name=\"Aperture\" GridType=\"Collection\" "
        "CollectionType=\"Temporal\" >"
     << std::endl;
}

template <typename T>
void
hdf_exporter<T>::writeXMFStep(std::ofstream &fs, int step,
                              double time) {
  std::string dim_str;
  auto &mesh = grid->mesh();
  if (grid->dim() == 3) {
    dim_str = fmt::format("{} {} {}", mesh.dims[2], mesh.dims[1],
                          mesh.dims[0]);
  } else if (grid->dim() == 2) {
    dim_str = fmt::format("{} {}", mesh.dims[1], mesh.dims[0]);
  } else if (grid->dim() == 1) {
    dim_str = fmt::format("{} 1", mesh.dims[0]);
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
  } else if (grid->dim() == 1) {
    fs << "  <Topology Type=\"2DSMesh\" NumberOfElements=\"" << dim_str
       << "\"/>" << std::endl;
    fs << "  <Geometry GeometryType=\"X_Y\">" << std::endl;
  }
  fs << "    <DataItem Dimensions=\"" << dim_str
     << "\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">"
     << std::endl;
  fs << "      mesh.h5:x1" << std::endl;
  fs << "    </DataItem>" << std::endl;
  if (grid->dim() >= 2) {
    fs << "    <DataItem Dimensions=\"" << dim_str
       << "\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">"
       << std::endl;
    fs << "      mesh.h5:x2" << std::endl;
    fs << "    </DataItem>" << std::endl;
  }
  if (grid->dim() >= 3) {
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

  for (auto &f : dbFields1d) {
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
      for (int i = 0; i < 1; i++) {
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

template <typename T>
void
hdf_exporter<T>::writeXMFTail(std::ofstream &fs) {
  fs << "</Grid>" << std::endl;
  fs << "</Domain>" << std::endl;
  fs << "</Xdmf>" << std::endl;
}

template <typename T>
void
hdf_exporter<T>::writeXMF(uint32_t step, double time) {
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

template <typename T>
void
hdf_exporter<T>::prepareXMFrestart(uint32_t restart_step,
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
