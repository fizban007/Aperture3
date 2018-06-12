#ifndef _GRID_H_
#define _GRID_H_

#include <array>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
// #include <functional>
#include "constant_defs.h"
#include "data/multi_array.h"
#include "data/quadmesh.h"
#include "data/typedefs.h"
#include "data/vec3.h"
// #include "metrics.h"

namespace Aperture {

// Currently the class Grid is simply a thin wrapper around Quadmesh.
class Grid {
 public:
  Grid();
  Grid(int N1, int N2 = 1, int N3 = 1);
  Grid(const Grid& g);
  Grid(Grid&& g);
  ~Grid();

  Grid& operator=(const Grid& g);
  Grid& operator=(Grid&& g);
  bool operator==(const Grid& g) const { return (m_mesh == g.m_mesh); }

  Quadmesh& mesh() { return m_mesh; }
  const Quadmesh& mesh() const { return m_mesh; }
  int size() const { return m_mesh.size(); }
  Extent extent() const { return m_mesh.extent(); }
  unsigned int dim() const { return m_mesh.dim(); }

 private:
  // void allocate_arrays();
  Quadmesh m_mesh;
};
}

// #include "data/detail/grid_impl.hpp"

#endif  // _GRID_H_
