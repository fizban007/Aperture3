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
#include "sim_params.h"
// #include "metrics.h"

namespace Aperture {

// Currently the class Grid is simply a thin wrapper around Quadmesh with some
// cached grid quantities
class Grid {
 public:
  Grid();
  Grid(int N1, int N2 = 1, int N3 = 1);
  Grid(const Grid& g);
  Grid(Grid&& g);
  ~Grid();

  void init(const SimParams& params);

  Grid& operator=(const Grid& g);
  Grid& operator=(Grid&& g);
  bool operator==(const Grid& g) const { return (m_mesh == g.m_mesh); }

  Quadmesh& mesh() { return m_mesh; }
  const Quadmesh& mesh() const { return m_mesh; }
  Quadmesh* mesh_ptr() { return &m_mesh; }
  const Quadmesh* mesh_ptr() const { return &m_mesh; }
  int size() const { return m_mesh.size(); }
  Extent extent() const { return m_mesh.extent(); }
  unsigned int dim() const { return m_mesh.dim(); }

  Scalar* D1() { return m_D1.data(); }
  Scalar* D2() { return m_D2.data(); }
  Scalar* D3() { return m_D3.data(); }
  Scalar* alpha_grr() { return m_alpha_grr.data(); }
  Scalar* A() { return m_A.data(); }
  Scalar* a2() { return m_a2.data(); }
  Scalar* angle() { return m_angle.data(); }

  struct mesh_ptrs {
    Scalar *D1, *D2, *D3;
    Scalar *A, *alpha_grr, *angle;
    Scalar *a2;
  };
  struct const_mesh_ptrs {
    const Scalar *D1, *D2, *D3;
    const Scalar *A, *alpha_grr, *angle;
    const Scalar *a2;
  };
  const_mesh_ptrs get_mesh_ptrs() const;

 private:
  // void allocate_arrays();
  Quadmesh m_mesh;

  MultiArray<Scalar> m_D1, m_D2, m_D3;
  MultiArray<Scalar> m_A, m_alpha_grr, m_angle;
  MultiArray<Scalar> m_a2;
};
}

// #include "data/detail/grid_impl.hpp"

#endif  // _GRID_H_
