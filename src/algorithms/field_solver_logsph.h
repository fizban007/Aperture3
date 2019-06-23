#ifndef _FIELD_SOLVER_LOGSPH_H_
#define _FIELD_SOLVER_LOGSPH_H_

#include "grids/grid_log_sph.h"

namespace Aperture {

struct sim_data;
class sim_environment;

class field_solver_logsph {
 public:
  field_solver_logsph(sim_environment& env);
  ~field_solver_logsph();

  void update_fields(sim_data& data, double dt, double time = 0.0);
  void apply_boundary(sim_data& data, double omega, double time = 0.0);

 private:
  sim_environment& m_env;
};

}  // namespace Aperture


#endif  // _FIELD_SOLVER_LOGSPH_H_
