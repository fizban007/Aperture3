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

