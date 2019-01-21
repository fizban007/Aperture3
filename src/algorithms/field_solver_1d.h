#ifndef _FIELD_SOLVER_1D_H_
#define _FIELD_SOLVER_1D_H_

#include "field_solver.h"

namespace Aperture {

class field_solver_1d : public field_solver {
 public:
  field_solver_1d();
  virtual ~field_solver_1d();
  virtual void update_fields(sim_data& data, double dt,
                             double time = 0.0) override;

  void update_fields(vfield_t& E, const vfield_t& J,
                     const vfield_t& J_bg, double dt,
                     double time = 0.0);
};  // ----- end of class field_solver_1d : public field_solver -----

}  // namespace Aperture

#endif  // _FIELD_SOLVER_1D_H_
