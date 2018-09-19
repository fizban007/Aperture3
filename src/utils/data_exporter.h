#ifndef _DATA_EXPORTER_H_
#define _DATA_EXPORTER_H_

#include <array>
#include <iostream>
#include <string>
#include <vector>
// #include <mutex>
#include "data/grid.h"
#include "data/vec3.h"
#include <silo.h>

namespace Aperture {

////////////////////////////////////////////////////////////////////////////////
///  Struct that contains the definition of a quadmesh object in Silo
///  hdf5.
////////////////////////////////////////////////////////////////////////////////
struct silo_quadMesh {
  std::string meshName;         //!< Name of the quadmesh
  int numDim;                   //!< Number of dimensions
  std::vector<int> sizeOfDims;  //!< Size of each dimension
  // int *sizeOfDims;                            //!< Size of each
  // dimension
  std::vector<std::vector<float>>
      gridPoints;  //!< Array that contains location of all the points
  float* gridArray[3];  //!< Pointer to the array that contains location
                        //!< of all
                        // the points
  bool linear;          //!< If the mesh is a colinear mesh
  std::array<std::string, 3> grid_conf;  //!< config strings of the grid

  silo_quadMesh() {
    for (int i = 0; i < 3; i++) {
      gridArray[i] = nullptr;
    }
  }

  silo_quadMesh(silo_quadMesh&& other)
      : meshName(other.meshName),
        numDim(other.numDim),
        sizeOfDims(std::move(other.sizeOfDims)),
        gridPoints(std::move(other.gridPoints)),
        linear(other.linear),
        grid_conf(std::move(other.grid_conf)) {
    for (int i = 0; i < 3; i++) {
      if (i < numDim)
        gridArray[i] = gridPoints[i].data();
      else
        gridArray[i] = nullptr;
    }
  }
};  // ----------  end of struct quadMesh  ----------

////////////////////////////////////////////////////////////////////////////////
///  Struct that contains a quadVariable object for Silo hdf5.
////////////////////////////////////////////////////////////////////////////////
template <typename FloatT>
struct silo_quadVariable {
  std::string quadName;  //!< Name of the quadvariable
  const FloatT* data;    //!< Pointer to the data
  std::string meshName;  //!< Name of the associated quadmesh
  silo_quadMesh* mesh;   //!< Pointer to the corresponding mesh
};  // ----------  end of struct quadVariable  ----------

////////////////////////////////////////////////////////////////////////////////
///  Struct that contains a data object for Silo hdf5, that is
///  unassociated with any mesh.
////////////////////////////////////////////////////////////////////////////////
struct silo_dbVariable {
  std::string varName;  //!< Name of the single variable
  float* data;          //!< Pointer to the data
};  // ----------  end of struct dbVariable  ----------

////////////////////////////////////////////////////////////////////////////////
///  Struct that contains an array object for Silo hdf5, that is
///  unassociated with any mesh.
////////////////////////////////////////////////////////////////////////////////
struct silo_dbArray {
  std::string varName;    //!< Name of the single variable
  std::vector<int> dims;  //!< Dimensions of the array
  int ndims;              //!< Number of array dimensions
  float* data;            //!< Pointer to the data
};  // ----------  end of struct dbVariable  ----------

////////////////////////////////////////////////////////////////////////////////
/// Class for communicating with output data files.
/// The object should be initialized with an output directory, a file
/// prefix format, and a default numbering scheme.
///
/// The object should also be initialized with a quadmesh, and the field
/// names that are defined on the mesh. The fields are the quantities
/// that get written to the output file. The mesh should be in float
/// format, with a name, array of dimensions, number of dimensions,
/// array of grid point locations.
///
/// Can select whether to use double as output format(TODO: implement
/// this feature). The default is to use double, but turning it off can
/// save a lot of disk space. Can also select whether to use
/// compression, which is on by default.
///
/// TODO: Finish documentation of this class
////////////////////////////////////////////////////////////////////////////////
class DataExporter {
 public:
  DataExporter();  //!< default constructor
  DataExporter(const std::string& dir, const std::string& prefix,
               bool compress, int rank = 0,
               int size = 1);  //!< constructor
  ~DataExporter();

  DataExporter& operator=(DataExporter&& other);  //!< Move assignment

  void SetDir(const char* dirName) {
    outputDirectory = std::string(dirName);
  }
  const std::string& GetDir() const { return outputDirectory; }
  void WriteOutput(
      int timeStep, float time, const Index& pos = Index(0, 0, 0),
      bool displayGuard = false);  //!< Writes output to the file
  void AddField(
      const char* name, const float* data,
      const char* mesh);  //!< Add a field with associated name and
                          // data pointer and mesh
  void AddField(
      const char* name, const double* data,
      const char* mesh);  //!< Add a field with associated name and
                          // data pointer and mesh
  void AddVariable(const char* name, float* data);
  void AddArray(const char* name, float* data, int* dims, int ndims);

  struct AddMesh_t {
    template <typename Metric>
    void operator()(const Metric& g, const char* meshName, int dim,
                    const Grid& grid, DataExporter& exporter);
  } AddMesh;

  void CopyConfig(const std::string& file);
  void CopyMain();

 private:
  void SetDefault();

  std::string
      outputDirectory;  //!< Sets the directory of all the data files
  std::string subDirectory;  //!< Sets the directory of current rank
  std::string subName;
  std::string filePrefix;  //!< Sets the common prefix of the data files

  std::vector<silo_quadMesh> quadMeshes;  //!< All the quadmesh objects
  std::vector<silo_quadVariable<float>>
      quadVarsF;  //!< All the quadVar objects
  std::vector<silo_quadVariable<double>>
      quadVarsD;                        //!< All the quadVar objects
  std::vector<silo_dbVariable> dbVars;  //!< All the dbVar objects
  std::vector<silo_dbArray> dbArrays;   //!< All the dbArray objects

  int myRank;
  int numRanks;
  int numFields;              //!< Caching the total number of fields
  int *lowOffset, *hiOffset;  //!< Lower and Upper ghost zones
  // bool useDouble;                         //!< Whether to use double
  // for data writes, false to save disk space
  bool useCompression;  //!< Whether to use compression in Silo files

};  // -----  end of class DataExporter  -----

}  // namespace Aperture

#endif  // _DATA_EXPORTER_H_

#include "utils/data_exporter_impl.hpp"
