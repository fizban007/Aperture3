#include "utils/hdf_exporter.h"
#include "fmt/ostream.h"
#include "utils/logger.h"
// #include "config_file.h"
#include "commandline_args.h"
#include <H5Cpp.h>
#include <boost/filesystem.hpp>
#include <time.h>
#include <fstream>
#include "nlohmann/json.hpp"

using json = nlohmann::json;

namespace Aperture {

// DataExporter::DataExporter() {}

DataExporter::DataExporter(const Grid& g, const std::string& dir, const std::string& prefix)
    : outputDirectory(dir), filePrefix(prefix), grid(g) {
  boost::filesystem::path rootPath (dir.c_str());
  boost::system::error_code returnedError;

  boost::filesystem::create_directories(rootPath, returnedError);
  if (outputDirectory.back() != '/') outputDirectory.push_back('/');

  // Format the output directory as Data%Y%m%d-%H%M
  char myTime[150] = {};
  char subDir[200] = {};
  time_t rawtime;
  struct tm* timeinfo;
  time(&rawtime);
  timeinfo = localtime(&rawtime);
  strftime(myTime, 140, "%Y%m%d-%H%M", timeinfo);
  snprintf(subDir, sizeof(subDir), "Data%s/", myTime);

  outputDirectory += subDir;
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

void
DataExporter::AddArray(const std::string &name, float *data, int *dims, int ndims) {
  dataset<float> tempData;


  tempData.name = name;
  tempData.data = data;
  tempData.ndims = ndims;
  for (int i = 0; i < ndims; i++)
    tempData.dims.push_back(dims[i]);

  dbFloat.push_back(tempData);
}

void
DataExporter::AddArray(const std::string &name, double *data, int *dims, int ndims) {
  dataset<double> tempData;

  tempData.name = name;
  tempData.data = data;
  tempData.ndims = ndims;
  for (int i = 0; i < ndims; i++)
    tempData.dims.push_back(dims[i]);

  dbDouble.push_back(tempData);
}

template <typename T>
void
DataExporter::AddArray(const std::string &name, MultiArray<T> &array) {
  int ndims = array.dim();
  int *dims = new int[ndims];
  for (int i = 0; i < ndims; i++)
    dims[i] = array.extent()[i];

  AddArray(name, array.data(), dims, ndims);

  delete[] dims;
}

template <typename T>
void
DataExporter::AddArray(const std::string &name, VectorField<T> &field, int component) {
  AddArray(name, field.data(component));
}

void
DataExporter::AddParticleArray(const std::string& name, const Particles& ptc) {
  ptcdata<Particles> temp;
  temp.name = name;
  temp.ptc = &ptc;
  temp.data_x = std::vector<float>(MAX_TRACKED);
  temp.data_p = std::vector<float>(MAX_TRACKED);

  dbPtcData.push_back(std::move(temp));
}

void
DataExporter::AddParticleArray(const std::string& name, const Photons& ptc) {
  ptcdata<Photons> temp;
  temp.name = name;
  temp.ptc = &ptc;
  temp.data_x = std::vector<float>(MAX_TRACKED);
  temp.data_p = std::vector<float>(MAX_TRACKED);
  temp.data_l = std::vector<float>(MAX_TRACKED);

  dbPhotonData.push_back(std::move(temp));
}

void
DataExporter::WriteOutput(int timestep, float time) {
  if (!checkDirectories()) createDirectories();
  try {
    std::string filename = outputDirectory + filePrefix + fmt::format("{0:06d}.h5", timestep);
    H5::H5File *file = new H5::H5File(filename, H5F_ACC_TRUNC);

    for (auto& ds : dbFloat) {
      hsize_t* sizes = new hsize_t[ds.ndims];
      for (int i = 0; i < ds.ndims; i++)
        sizes[i] = ds.dims[i];
      H5::DataSpace space(ds.ndims, sizes);
      H5::DataSet *dataset = new H5::DataSet(file->createDataSet(ds.name, H5::PredType::NATIVE_FLOAT, space));
      dataset->write((void*)ds.data, H5::PredType::NATIVE_FLOAT);

      delete[] sizes;
      delete dataset;
    }

    for (auto& ds : dbDouble) {
      hsize_t* sizes = new hsize_t[ds.ndims];
      for (int i = 0; i < ds.ndims; i++)
        sizes[i] = ds.dims[i];
      H5::DataSpace space(ds.ndims, sizes);
      H5::DataSet *dataset = new H5::DataSet(file->createDataSet(ds.name, H5::PredType::NATIVE_DOUBLE, space));
      dataset->write((void*)ds.data, H5::PredType::NATIVE_DOUBLE);

      delete[] sizes;
      delete dataset;
    }

    for (auto& ds : dbPtcData) {
      std::string name_x = ds.name + "_x";
      std::string name_p = ds.name + "_p";
      unsigned int idx = 0;
      for (Index_t n = 0; n < ds.ptc->number(); n++) {
        if (!ds.ptc->is_empty(n) && ds.ptc->check_flag(n, ParticleFlag::tracked) && idx < MAX_TRACKED) {
          Scalar x = grid.mesh().pos(0, ds.ptc->data().cell[n], ds.ptc->data().x1[n]);
          ds.data_x[idx] = x;
          ds.data_p[idx] = ds.ptc->data().p1[n];
          idx += 1;
        }
      }
      hsize_t sizes[1] = { idx };
      H5::DataSpace space(1, sizes);
      H5::DataSet *dataset_x = new H5::DataSet(file->createDataSet(name_x, H5::PredType::NATIVE_FLOAT, space));
      dataset_x->write((void*)ds.data_x.data(), H5::PredType::NATIVE_FLOAT);
      H5::DataSet *dataset_p = new H5::DataSet(file->createDataSet(name_p, H5::PredType::NATIVE_FLOAT, space));
      dataset_p->write((void*)ds.data_p.data(), H5::PredType::NATIVE_FLOAT);

      delete dataset_x;
      delete dataset_p;

      Logger::print_info("Written {} tracked particles", idx);
    }

    for (auto& ds : dbPhotonData) {
      std::string name_x = ds.name + "_x";
      std::string name_p = ds.name + "_p";
      // std::string name_l = ds.name + "_l";
      unsigned int idx = 0;
      for (Index_t n = 0; n < ds.ptc->number(); n++) {
        if (!ds.ptc->is_empty(n) && ds.ptc->check_flag(n, PhotonFlag::tracked) && idx < MAX_TRACKED) {
          Scalar x = grid.mesh().pos(0, ds.ptc->data().cell[n], ds.ptc->data().x1[n]);
          ds.data_x[idx] = x;
          ds.data_p[idx] = ds.ptc->data().p1[n];
          // ds.data_l[idx] = ds.ptc->data().path[n];
          idx += 1;
        }
      }
      hsize_t sizes[1] = { idx };
      H5::DataSpace space(1, sizes);
      H5::DataSet *dataset_x = new H5::DataSet(file->createDataSet(name_x, H5::PredType::NATIVE_FLOAT, space));
      dataset_x->write((void*)ds.data_x.data(), H5::PredType::NATIVE_FLOAT);
      H5::DataSet *dataset_p = new H5::DataSet(file->createDataSet(name_p, H5::PredType::NATIVE_FLOAT, space));
      dataset_p->write((void*)ds.data_p.data(), H5::PredType::NATIVE_FLOAT);
      // H5::DataSet *dataset_l = new H5::DataSet(file->createDataSet(name_l, H5::PredType::NATIVE_FLOAT, space));
      // dataset_l->write((void*)ds.data_l.data(), H5::PredType::NATIVE_FLOAT);

      delete dataset_x;
      delete dataset_p;
      // delete dataset_l;

      Logger::print_info("Written {} tracked photons", idx);
    }
    delete file;
  }
  // catch failure caused by the H5File operations
  catch( H5::FileIException &error )
  {
    error.printErrorStack();
    // return -1;
  }
  // catch failure caused by the DataSet operations
  catch( H5::DataSetIException &error )
  {
    error.printErrorStack();
    // return -1;
  }
  // catch failure caused by the DataSpace operations
  catch( H5::DataSpaceIException &error )
  {
    error.printErrorStack();
    // return -1;
  }

}

void
DataExporter::writeConfig(const SimParams& params) {
  if (!checkDirectories()) createDirectories();
  std::string filename = outputDirectory + "config.json";
  auto& c = params;
  json conf = {
    {"delta_t", c.delta_t},
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
    {"e_min", c.e_min}
  };

  std::ofstream o(filename);
  o << std::setw(4) << conf << std::endl;
}


// Explicit instantiation of templates
template
void
DataExporter::AddArray<float>(const std::string &name, MultiArray<float> &array);

template
void
DataExporter::AddArray<double>(const std::string &name, MultiArray<double> &array);

template
void
DataExporter::AddArray<float>(const std::string &name, VectorField<float> &field, int component);

template
void
DataExporter::AddArray<double>(const std::string &name, VectorField<double> &field, int component);
}
