#include "data/grid.h"
// #include "CudaLE.h"
// #include "algorithms/interpolation.h"
// #include <Eigen/Dense>
#include <cctype>
#include <iomanip>
#include <iostream>
#include <utility>

using namespace Aperture;

Grid::Grid() :
    Grid(1, 1, 1) {}

Grid::~Grid() {}

Grid::Grid(int N1, int N2, int N3)
    : m_mesh(N1, N2, N3) {}


Grid::Grid(const Grid& g) {
  m_mesh = g.m_mesh;
}

Grid::Grid(Grid&& g) {
  m_mesh = g.m_mesh;
}

Grid&
Grid::operator=(const Grid& g) {
  m_mesh = g.m_mesh;
  return *this;
}

Grid&
Grid::operator=(Grid&& g) {
  m_mesh = g.m_mesh;
  return *this;
}

void
Grid::init(const SimParams& params) {
  // Setup the mesh
  for (int i = 0; i < 3; i++) {
    m_mesh.guard[i] = params.guard[i];
    m_mesh.sizes[i] = params.size[i];
    m_mesh.lower[i] = params.lower[i];
    m_mesh.dims[i] = params.N[i] + 2 * params.guard[i];
    m_mesh.delta[i] = params.size[i] / params.N[i];
    m_mesh.tileSize[i] = params.tile_size[i];
    m_mesh.dimension = m_mesh.dim();
  }

  // TODO: In the near future, cache all the metric terms here
}