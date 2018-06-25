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
  }
  m_mesh.dimension = m_mesh.dim();

  if (m_mesh.dims[0] * m_mesh.dims[1] * m_mesh.dims[2] >= MAX_CELL - 1) {
    std::cout << "Grid dimensions too large!" << std::endl;
    abort();
  }

  // TODO: In the near future, cache all the metric terms here
  // Initialize D1, D2, D3, alpha_grr, and A
  m_D1.resize(m_mesh.dims[0]); m_D1.assign(0.0);
  m_D2.resize(m_mesh.dims[0]); m_D2.assign(1.0);
  m_D3.resize(m_mesh.dims[0]); m_D3.assign(0.0);
  m_alpha_grr.resize(m_mesh.dims[0]); m_alpha_grr.assign(1.0);
  m_A.resize(m_mesh.dims[0]); m_A.assign(1.0);
  m_a2.resize(m_mesh.dims[0]); m_a2.assign(1.0);
  m_angle.resize(m_mesh.dims[0]); m_angle.assign(0.0);
}

Grid::const_mesh_ptrs
Grid::get_mesh_ptrs() const {
  const_mesh_ptrs result;
  result.D1 = m_D1.data();
  result.D2 = m_D2.data();
  result.D3 = m_D3.data();
  result.alpha_grr = m_alpha_grr.data();
  result.A = m_A.data();
  result.a2 = m_a2.data();
  result.angle = m_angle.data();
  return result;
}
