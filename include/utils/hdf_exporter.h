#ifndef _HDF_EXPORTER_H_
#define _HDF_EXPORTER_H_

// #include <hdf5.h>
// #include <H5Cpp.h>
#include <string>
#include <vector>
#include "data/fields.h"
#include "data/particles.h"
#include "data/photons.h"

namespace Aperture {

template <typename T>
struct dataset {
  std::string name;
  std::vector<int> dims;
  int ndims;
  T* data;
};

template <typename Ptc>
struct ptcdata {
  std::string name;
  const Ptc* ptc;
  std::vector<float> data_x;
  std::vector<float> data_p;
};

class DataExporter
{
 public:
  DataExporter();
  DataExporter(const std::string& dir, const std::string& prefix);

  ~DataExporter();

  void WriteOutput(int timestep, float time);

  void AddArray(const std::string& name, float* data, int* dims, int ndims);
  void AddArray(const std::string& name, double* data, int* dims, int ndims);
  template <typename T>
  void AddArray(const std::string& name, VectorField<T>& field, int component);
  template <typename T>
  void AddArray(const std::string& name, MultiArray<T>& field);

  void AddParticleArray(const std::string& name, const Particles& ptc);
  void AddParticleArray(const std::string& name, const Photons& ptc);
  // void AddParticleArray(const Photons& ptc);

  void CopyConfig(const std::string& file);
  void CopyMain();

  void setGrid(const Grid& g) { grid = g; }

 private:
  std::string outputDirectory;  //!< Sets the directory of all the data files
  std::string subDirectory;     //!< Sets the directory of current rank
  std::string subName;
  std::string filePrefix;       //!< Sets the common prefix of the data files

  std::vector<dataset<float>> dbFloat;
  std::vector<dataset<double>> dbDouble;

  std::vector<ptcdata<Particles>> dbPtcData;
  std::vector<ptcdata<Photons>> dbPhotonData;

  Grid grid;
}; // ----- end of class DataExporter -----


}


#endif  // _HDF_EXPORTER_H_
