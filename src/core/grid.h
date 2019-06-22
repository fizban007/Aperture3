#ifndef _GRID_H_
#define _GRID_H_

#include <array>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "core/constant_defs.h"
#include "core/quadmesh.h"
#include "core/typedefs.h"
#include "core/vec3.h"
#include "sim_params.h"

namespace Aperture {

/// The base class Grid is simply a thin wrapper around Quadmesh.  Any
/// grid in specific coordinate systems should derive from this class
/// and implement the `init` method.
class Grid {
 public:
  Grid();
  Grid(int N1, int N2 = 1, int N3 = 1);
  Grid(const Grid& g);
  Grid(Grid&& g);
  virtual ~Grid();

  /// Initialize the grid parameters
  void init(const SimParams& params);
  virtual void compute_coef();

  Grid& operator=(const Grid& g);
  Grid& operator=(Grid&& g);
  bool operator==(const Grid& g) const { return (m_mesh == g.m_mesh); }

  Quadmesh& mesh() { return m_mesh; }
  const Quadmesh& mesh() const { return m_mesh; }
  int size() const { return m_mesh.size(); }
  Extent extent() const { return m_mesh.extent(); }

  /// Dimension of the grid
  unsigned int dim() const { return m_mesh.dim(); }

 protected:
  Quadmesh m_mesh;
};

}  // namespace Aperture

#endif  // _GRID_H_
