#include "utils/hdf_exporter_parallel.h"
#include "commandline_args.h"
#include "config_file.h"
#include "fmt/ostream.h"
#include "highfive/H5DataSet.hpp"
#include "highfive/H5DataSpace.hpp"
#include "highfive/H5File.hpp"
#include "nlohmann/json.hpp"
#include "utils/logger.h"
#include <H5Cpp.h>
#include <boost/filesystem.hpp>
#include <fstream>
#include <mpi.h>
#include <time.h>

using json = nlohmann::json;
using namespace HighFive;

namespace Aperture {

DataExporterParallel::DataExporterParallel() {}

DataExporterParallel::DataExporterParallel(const DomainInfo &info,
                                           const std::string &dir,
                                           const std::string &prefix)
    : outputDirectory(dir), filePrefix(prefix) {
  if (info.rank == 0) {
    boost::filesystem::path rootPath(dir.c_str());
    boost::system::error_code returnedError;

    boost::filesystem::create_directories(rootPath, returnedError);
    if (outputDirectory.back() != '/') outputDirectory.push_back('/');

    // Format the output directory as Data%Y%m%d-%H%M
    char myTime[100] = {};
    char subDir[100] = {};
    time_t rawtime;
    struct tm *timeinfo;
    time(&rawtime);
    timeinfo = localtime(&rawtime);
    strftime(myTime, 100, "%Y%m%d-%H%M", timeinfo);
    snprintf(subDir, sizeof(subDir), "Data%s/", myTime);

    outputDirectory += subDir;
    boost::filesystem::path subPath(outputDirectory);

    boost::filesystem::create_directories(subPath, returnedError);
  }
}

DataExporterParallel::~DataExporterParallel() {}

void
DataExporterParallel::AddArray(const std::string &name, float *data,
                               int *dims, int ndims) {
  dataset<float> tempData;

  tempData.name = name;
  tempData.data = data;
  tempData.ndims = ndims;
  for (int i = 0; i < ndims; i++) tempData.dims.push_back(dims[i]);

  dbFloat.push_back(tempData);
}

void
DataExporterParallel::AddArray(const std::string &name, double *data,
                               int *dims, int ndims) {
  dataset<double> tempData;

  tempData.name = name;
  tempData.data = data;
  tempData.ndims = ndims;
  for (int i = 0; i < ndims; i++) tempData.dims.push_back(dims[i]);

  dbDouble.push_back(tempData);
}

template <typename T>
void
DataExporterParallel::AddArray(const std::string &name,
                               multi_array_dev<T> &array) {
  int ndims = array.dim();
  int *dims = new int[ndims];
  for (int i = 0; i < ndims; i++) dims[i] = array.extent()[i];

  AddArray(name, array.data(), dims, ndims);

  delete[] dims;
}

template <typename T>
void
DataExporterParallel::AddArray(const std::string &name,
                               VectorField<T> &field, int component) {
  AddArray(name, field.data(component));
}

void
DataExporterParallel::AddParticleArray(const std::string &name,
                                       const Particles &ptc) {
  ptcdata<Particles> temp;
  temp.name = name;
  temp.ptc = &ptc;
  temp.data_x = std::vector<float>(MAX_TRACKED);
  temp.data_p = std::vector<float>(MAX_TRACKED);

  dbPtcData.push_back(std::move(temp));
}

void
DataExporterParallel::AddParticleArray(const std::string &name,
                                       const Photons &ptc) {
  ptcdata<Photons> temp;
  temp.name = name;
  temp.ptc = &ptc;
  temp.data_x = std::vector<float>(MAX_TRACKED);
  temp.data_p = std::vector<float>(MAX_TRACKED);

  dbPhotonData.push_back(std::move(temp));
}

void
DataExporterParallel::WriteOutput(int timestep, float time) {
  try {
    std::string filename = outputDirectory + filePrefix +
                           fmt::format("{0:06d}.h5", timestep);
    // H5::H5File *file = new H5::H5File(filename, H5F_ACC_TRUNC);
    File file(filename, File::ReadWrite | File::Create | File::Truncate,
              MPIOFileDriver(MPI_COMM_WORLD, MPI_INFO_NULL));

    for (auto &ds : dbFloat) {
      hsize_t *sizes = new hsize_t[ds.ndims];
      for (int i = 0; i < ds.ndims; i++) sizes[i] = ds.dims[i];
      H5::DataSpace space(ds.ndims, sizes);
      H5::DataSet *dataset = new H5::DataSet(file->createDataSet(
          ds.name, H5::PredType::NATIVE_FLOAT, space));
      dataset->write((void *)ds.data, H5::PredType::NATIVE_FLOAT);

      delete[] sizes;
      delete dataset;
    }

    for (auto &ds : dbDouble) {
      hsize_t *sizes = new hsize_t[ds.ndims];
      for (int i = 0; i < ds.ndims; i++) sizes[i] = ds.dims[i];
      H5::DataSpace space(ds.ndims, sizes);
      H5::DataSet *dataset = new H5::DataSet(file->createDataSet(
          ds.name, H5::PredType::NATIVE_DOUBLE, space));
      dataset->write((void *)ds.data, H5::PredType::NATIVE_DOUBLE);

      delete[] sizes;
      delete dataset;
    }

    for (auto &ds : dbPtcData) {
      std::string name_x = ds.name + "_x";
      std::string name_p = ds.name + "_p";
      unsigned int idx = 0;
      for (Index_t n = 0; n < ds.ptc->number(); n++) {
        if (!ds.ptc->is_empty(n) &&
            ds.ptc->check_flag(n, ParticleFlag::tracked) &&
            idx < MAX_TRACKED) {
          Scalar x = grid.mesh().pos(0, ds.ptc->data().cell[n],
                                     ds.ptc->data().x1[n]);
          ds.data_x[idx] = x;
          ds.data_p[idx] = ds.ptc->data().p1[n];
          idx += 1;
        }
      }
      hsize_t sizes[1] = {idx};
      H5::DataSpace space(1, sizes);
      H5::DataSet *dataset_x = new H5::DataSet(file->createDataSet(
          name_x, H5::PredType::NATIVE_FLOAT, space));
      dataset_x->write((void *)ds.data_x.data(),
                       H5::PredType::NATIVE_FLOAT);
      H5::DataSet *dataset_p = new H5::DataSet(file->createDataSet(
          name_p, H5::PredType::NATIVE_FLOAT, space));
      dataset_p->write((void *)ds.data_p.data(),
                       H5::PredType::NATIVE_FLOAT);

      delete dataset_x;
      delete dataset_p;

      Logger::print_info("Written {} tracked particles", idx);
    }

    for (auto &ds : dbPhotonData) {
      std::string name_x = ds.name + "_x";
      std::string name_p = ds.name + "_p";
      unsigned int idx = 0;
      for (Index_t n = 0; n < ds.ptc->number(); n++) {
        if (!ds.ptc->is_empty(n) &&
            ds.ptc->check_flag(n, PhotonFlag::tracked) &&
            idx < MAX_TRACKED) {
          Scalar x = grid.mesh().pos(0, ds.ptc->data().cell[n],
                                     ds.ptc->data().x1[n]);
          ds.data_x[idx] = x;
          ds.data_p[idx] = ds.ptc->data().p1[n];
          idx += 1;
        }
      }
      hsize_t sizes[1] = {idx};
      H5::DataSpace space(1, sizes);
      H5::DataSet *dataset_x = new H5::DataSet(file->createDataSet(
          name_x, H5::PredType::NATIVE_FLOAT, space));
      dataset_x->write((void *)ds.data_x.data(),
                       H5::PredType::NATIVE_FLOAT);
      H5::DataSet *dataset_p = new H5::DataSet(file->createDataSet(
          name_p, H5::PredType::NATIVE_FLOAT, space));
      dataset_p->write((void *)ds.data_p.data(),
                       H5::PredType::NATIVE_FLOAT);

      delete dataset_x;
      delete dataset_p;

      Logger::print_info("Written {} tracked photons", idx);
    }
    delete file;
  }
  // catch failure caused by the H5File operations
  catch (H5::FileIException &error) {
    error.printError();
    // return -1;
  }
  // catch failure caused by the DataSet operations
  catch (H5::DataSetIException &error) {
    error.printError();
    // return -1;
  }
  // catch failure caused by the DataSpace operations
  catch (H5::DataSpaceIException &error) {
    error.printError();
    // return -1;
  }
}

void
DataExporterParallel::writeConfig(const ConfigFile &config,
                                  const CommandArgs &args) {
  std::string filename = outputDirectory + "config.json";
  auto &c = config.data();
  json conf = {{"delta_t", c.delta_t},
               {"q_e", c.q_e},
               {"ptc_per_cell", c.ptc_per_cell},
               {"ion_mass", c.ion_mass},
               {"boundary_periodic", c.boundary_periodic[0]},
               {"create_pairs", c.create_pairs},
               {"trace_photons", c.trace_photons},
               {"gamma_thr", c.gamma_thr},
               {"photon_path", c.photon_path},
               {"grid",
                {{"N", grid.mesh().dims[0]},
                 {"guard", grid.mesh().guard[0]},
                 {"lower", grid.mesh().lower[0]},
                 {"size", grid.mesh().sizes[0]}}},
               {"interp_order", c.interpolation_order},
               {"track_pct", c.track_percent},
               {"ic_path", c.ic_path},
               {"N_steps", args.steps()},
               {"data_interval", args.data_interval()},
               {"spectral_alpha", c.spectral_alpha},
               {"e_s", c.e_s},
               {"e_min", c.e_min}};

  std::ofstream o(filename);
  o << std::setw(4) << conf << std::endl;
}

// Explicit instantiation of templates
template void DataExporterParallel::AddArray<float>(
    const std::string &name, multi_array_dev<float> &array);

template void DataExporterParallel::AddArray<double>(
    const std::string &name, multi_array_dev<double> &array);

template void DataExporterParallel::AddArray<float>(
    const std::string &name, VectorField<float> &field, int component);

template void DataExporterParallel::AddArray<double>(
    const std::string &name, VectorField<double> &field, int component);
}  // namespace Aperture
