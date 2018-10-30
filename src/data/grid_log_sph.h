#ifndef _GRID_LOG_SPH_H_
#define _GRID_LOG_SPH_H_

#include "data/grid.h"

namespace Aperture {

class Grid_LogSph : public Grid
{
 public:
  Grid_LogSph();
  virtual ~Grid_LogSph();

  void init(const SimParams& params);

  struct mesh_ptrs {
    Scalar *h1, *h2, *h3;
  };

 private:
  MultiArray<Scalar> m_h1, m_h2, m_h3;
}; // ----- end of class Grid_LogSph : public Grid -----


}

#endif  // _GRID_LOG_SPH_H_
