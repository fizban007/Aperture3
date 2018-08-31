#ifndef _DATA_EXPORTER_IMPL_H_
#define _DATA_EXPORTER_IMPL_H_
#include "utils/data_exporter.h"
//#define BOOST_NO_CXX11_SCOPED_ENUMS
#include "boost/filesystem.hpp"

namespace Aperture {


template <typename Metric>
void
DataExporter::AddMesh_t::operator()(const Metric& g, const char* meshName, int dim, const Grid& grid, DataExporter &exporter) {
  bool linearMesh = Metric::isLinear;
  // If a coord system is axisymmetric then we swap y and z axis
  bool swapZY = Metric::swapZY;

  silo_quadMesh tempMesh;
  tempMesh.meshName = meshName;
  tempMesh.numDim = dim;
  tempMesh.gridPoints.resize(dim);
  for (int i = 0; i < dim; i++) {
    if (linearMesh)
      tempMesh.gridPoints[i].resize(grid.mesh().dims[i], 0.0);
    else
      tempMesh.gridPoints[i].resize(grid.mesh().size(), 0.0);
    tempMesh.sizeOfDims.push_back(grid.mesh().dims[i]);
    tempMesh.gridArray[i] = tempMesh.gridPoints[i].data();
    exporter.lowOffset[i] = grid.mesh().guard[i] - 1;
    exporter.hiOffset[i] = grid.mesh().guard[i] - 1;
  }
  tempMesh.linear = linearMesh;
  tempMesh.grid_conf = grid.gen_config();

  if (linearMesh) {
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < grid.mesh().dims[i]; ++j) {
        tempMesh.gridPoints[i][j] = grid.mesh().pos(i, j, 0);
      }
    }
  } else {
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < grid.mesh().size(); ++j) {
        Index idx(j, grid.mesh().extent());
        Vec3<Scalar> pos(grid.mesh().pos(0, idx.x, 1),
                         (grid.dim() > 1 ? grid.mesh().pos(1, idx.y, 1) : 0.0),
                         (grid.dim() > 2 ? grid.mesh().pos(2, idx.z, 1) : 0.0));

        g.PosToCartesian(pos);
        if (swapZY) {
          std::swap(pos[1], pos[2]);
        }

        tempMesh.gridPoints[i][j] = pos[i];
      }
    }
  }

  boost::filesystem::path rootPath(exporter.outputDirectory.c_str());
  boost::system::error_code returnedError;

  if (!boost::filesystem::exists(rootPath)) {
    boost::filesystem::create_directories(rootPath, returnedError);
    if (returnedError) std::cerr << "Error creating directory!" << std::endl;
  }
  char filename[50];
  sprintf(filename, "%sgrid_config", exporter.outputDirectory.c_str());

  std::cout << filename << std::endl;
  std::ofstream fs;
  fs.open(filename, std::ios::out);
  for (unsigned int i = 0; i < tempMesh.grid_conf.size(); i++) {
    fs << tempMesh.grid_conf[i] << std::endl;
  }
  fs.close();

  exporter.quadMeshes.push_back(std::move(tempMesh));
}

}

#endif  // _DATA_EXPORTER_IMPL_H_
