#ifndef _FIELD_SOLVER_1DGR_H_
#define _FIELD_SOLVER_1DGR_H_

#include "grids/grid_1dgr.h"

namespace Aperture {

struct sim_data;
class sim_environment;

class field_solver_1dgr {
 public:
  field_solver_1dgr(sim_environment& env);
  ~field_solver_1dgr();

  void update_fields(sim_data& data, double dt, double time = 0.0);

 private:
  sim_environment& m_env;
};

}  // namespace Aperture



#endif  // _FIELD_SOLVER_1DGR_H_
