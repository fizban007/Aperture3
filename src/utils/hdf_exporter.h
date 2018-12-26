#ifndef _HDF_EXPORTER_H_
#define _HDF_EXPORTER_H_

// #include <hdf5.h>
// #include <H5Cpp.h>
#include "data/fields.h"
#include "data/grid.h"
#include "data/particles.h"
#include "data/photons.h"
#include "sim_params.h"
#include <string>
#include <vector>
#include <boost/multi_array.hpp>
#include <fstream>
#include <memory>

namespace Aperture {

class ConfigFile;
class CommandArgs;
struct SimParams;
class Environment;
struct SimData;

template <typename T>
struct dataset {
  std::string name;
  std::vector<int> dims;
  int ndims;
  T* data;
};

template <typename Ptc>
struct ptcoutput {
  std::string name;
  const Ptc* ptc;
  std::vector<float> data_x;
  std::vector<float> data_p;
};

template <>
struct ptcoutput<Photons> {
  std::string name;
  const Photons* ptc;
  std::vector<float> data_x;
  std::vector<float> data_p;
  std::vector<float> data_l;
};

// template <typename T, int n>
// struct vfieldoutput {
//   std::string name;
//   VectorField<T>* field;
//   boost::multi_array<float, n> f1;
//   boost::multi_array<float, n> f2;
//   boost::multi_array<float, n> f3;
//   bool sync;
// };

template <int n>
struct fieldoutput {
  std::string name;
  std::string type;
  // ScalarField<T>* field;
  field_base* field;
  std::vector<boost::multi_array<float, n>> f;
  bool sync;
};

class DataExporter {
 public:
  // DataExporter();
  DataExporter(Environment& env, SimData& data, uint32_t& timestep);

  ~DataExporter();

  void WriteGrid();

  void WriteOutput(int timestep, double time);

  void AddArray(const std::string& name, float* data, int* dims,
                int ndims);
  void AddArray(const std::string& name, double* data, int* dims,
                int ndims);
  template <typename T>
  void AddArray(const std::string& name, VectorField<T>& field,
                int component);
  template <typename T>
  void AddArray(const std::string& name, multi_array_dev<T>& field);

  template <typename T>
  void AddField(const std::string& name, ScalarField<T>& field, bool sync = true);
  template <typename T>
  void AddField(const std::string& name, VectorField<T>& field, bool sync = true);

  template <typename T>
  void InterpolateFieldValues(fieldoutput<2>& field, int components, T t);
  template <typename T>
  void InterpolateFieldValues(fieldoutput<3>& field, int components, T t);

  void AddParticleArray(const std::string& name, const Particles& ptc);
  void AddParticleArray(const std::string& name, const Photons& ptc);
  // void AddParticleArray(const Photons& ptc);

  // void CopyConfig(const std::string& file);
  // void CopyMain();

  // void setGrid(const Grid& g) { grid = g; }
  void writeConfig(const SimParams& params);
  void writeXMFHead(std::ofstream& fs);
  void writeXMFStep(std::ofstream& fs, int step, double time);
  void writeXMFTail(std::ofstream& fs);
  void writeXMF(uint32_t step, double time);
  void prepareXMFrestart(uint32_t restart_step, int data_interval);
  void createDirectories();
  bool checkDirectories();
  void copyConfigFile();
  void copySrc();

  void writeSnapshot(Environment& env, SimData& data, uint32_t timestep);
  void load_from_snapshot(Environment& env, SimData& data, uint32_t& timestep);

 private:
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
  std::vector<fieldoutput<2>> dbFields2d;
  std::vector<fieldoutput<3>> dbFields3d;
  // meshoutput dbMesh;

  std::vector<ptcoutput<Particles>> dbPtcData;
  std::vector<ptcoutput<Photons>> dbPhotonData;

  // Grid grid;
  std::unique_ptr<Grid> grid;
  int downsample_factor;
  std::ofstream xmf;
};  // ----- end of class DataExporter -----

}  // namespace Aperture

#endif  // _HDF_EXPORTER_H_
