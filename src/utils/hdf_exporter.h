#ifndef _HDF_EXPORTER_H_
#define _HDF_EXPORTER_H_

// #include <hdf5.h>
// #include <H5Cpp.h>
// #include "data/fields_dev.h"
#include "core/grid.h"
#include "core/multi_array.h"
#include "data/field_base.h"
#include "data/particle_interface.h"
// #include "data/particles_dev.h"
// #include "data/photons_dev.h"
#include "sim_params.h"
#include <boost/multi_array.hpp>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace Aperture {

class ConfigFile;
class CommandArgs;
struct SimParams;
// class Photons;

template <typename T>
struct dataset {
  std::string name;
  std::vector<int> dims;
  int ndims;
  T* data;
};

template <int n>
struct fieldoutput {
  std::string name;
  std::string type;
  field_base* field;
  std::vector<boost::multi_array<float, n>> f;
  bool sync;
};

template <typename T, int n>
struct arrayoutput {
  std::string name;
  multi_array<T>* array;
  boost::multi_array<float, n> f;
};

struct ptcoutput {
  std::string name;
  std::string type;
  particle_interface* ptc;
};

struct ptcoutput_1d {
  std::string name;
  std::string type;
  particle_interface* ptc;
  std::vector<float> x;
  std::vector<float> p;
};

struct ptcoutput_2d {
  std::string name;
  std::string type;
  particle_interface* ptc;
  std::vector<float> x1;
  std::vector<float> x2;
  std::vector<float> x3;
  std::vector<float> p1;
  std::vector<float> p2;
  std::vector<float> p3;
};

// Using the CRTP to allow base class to access derived class methods
template <typename DerivedClass>
class hdf_exporter {
 public:
  // hdf_exporter();
  hdf_exporter(SimParams& params, uint32_t& timestep);
  virtual ~hdf_exporter();

  void WriteGrid();
  void WriteOutput(int timestep, double time);

  void writeConfig(const SimParams& params);
  void writeXMFHead(std::ofstream& fs);
  void writeXMFStep(std::ofstream& fs, uint32_t step, double time);
  void writeXMFTail(std::ofstream& fs);
  void writeXMF(uint32_t step, double time);
  void prepareXMFrestart(uint32_t restart_step, int data_interval);
  void createDirectories();
  bool checkDirectories();
  void copyConfigFile();
  void copySrc();

  void add_field_output(const std::string& name,
                        const std::string& type, int num_components,
                        field_base* field, int dim, bool sync = false);
  void add_ptc_output(const std::string& name, const std::string& type,
                      particle_interface* ptc);
  void add_ptc_output_1d(const std::string& name,
                         const std::string& type,
                         particle_interface* ptc);
  void add_ptc_output_2d(const std::string& name,
                         const std::string& type,
                         particle_interface* ptc);
  template <typename T>
  void add_array_output(const std::string& name, multi_array<T>& array);

  // void AddArray(const std::string& name, float* data, int* dims,
  //               int ndims);
  // void AddArray(const std::string& name, double* data, int* dims,
  //               int ndims);
  // template <typename T>
  // void AddArray(const std::string& name, cu_vector_field<T>&
  // field,
  //               int component);
  // template <typename T>
  // void AddArray(const std::string& name, cu_multi_array<T>&
  // field);

  // template <typename T>
  // void AddField(const std::string& name, cu_scalar_field<T>&
  // field, bool sync = true); template <typename T> void
  // AddField(const std::string& name, cu_vector_field<T>& field,
  // bool sync = true);

  // template <typename T>
  // void InterpolateFieldValues(fieldoutput<2>& field, int
  // components, T t); template <typename T> void
  // InterpolateFieldValues(fieldoutput<3>& field, int components, T
  // t);

  // void AddParticleArray(const std::string& name, const Particles&
  // ptc); void AddParticleArray(const std::string& name, const
  // Photons& ptc); void AddParticleArray(const Photons& ptc);

  // void CopyConfig(const std::string& file);
  // void CopyMain();

  // void setGrid(const Grid& g) { grid = g; }
 protected:
  std::string
      outputDirectory;  //!< Sets the directory of all the data files
  std::string subDirectory;  //!< Sets the directory of current rank
  std::string subName;
  std::string filePrefix;  //!< Sets the common prefix of the data files
  const SimParams& m_params;

  // std::vector<dataset<float>> dbFloat;
  // std::vector<dataset<double>> dbDouble;
  // std::vector<sfieldoutput2d<double>> dbScalars2d;
  // std::vector<vfieldoutput2d<double>> dbVectors2d;
  // std::vector<sfieldoutput3d<double>> dbScalars3d;
  // std::vector<vfieldoutput3d<double>> dbVectors3d;
  std::vector<fieldoutput<1>> dbFields1d;
  std::vector<fieldoutput<2>> dbFields2d;
  std::vector<fieldoutput<3>> dbFields3d;
  std::vector<arrayoutput<float, 2>> dbfloat2d;
  std::vector<ptcoutput> dbPtc;
  // meshoutput dbMesh;

  std::vector<ptcoutput_1d> dbPtcData1d;
  std::vector<ptcoutput_2d> dbPtcData2d;
  // std::vector<ptcoutput<Photons>> dbPhotonData;

  // Grid grid;
  std::unique_ptr<Grid> grid;
  std::unique_ptr<Grid> orig_grid;
  int downsample_factor;
  std::ofstream xmf;
};  // ----- end of class hdf_exporter -----

}  // namespace Aperture

#endif  // _HDF_EXPORTER_H_
