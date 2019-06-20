#ifndef _FIELD_SOLVER_1DGR_H_
#define _FIELD_SOLVER_1DGR_H_

#include "cuda/grids/grid_1dgr_dev.h"

namespace Aperture {

struct cu_sim_data1d;

class field_solver_1dgr_dev {
 public:
  field_solver_1dgr_dev();
  virtual ~field_solver_1dgr_dev();

  void update_fields(cu_sim_data1d& data, double dt, double time = 0.0);

 // private:
 //  const Grid_1dGR_dev& m_grid;
};  // ----- end of class field_solver_1dgr_dev -----

}  // namespace Aperture

#endif  // _FIELD_SOLVER_1DGR_H_
