#ifndef _FIELD_SOLVER_H_
#define _FIELD_SOLVER_H_

#include "data/fields.h"

namespace Aperture {

struct sim_data;

class field_solver
{
 public:
  typedef vector_field<Scalar> vfield_t;
  typedef scalar_field<Scalar> sfield_t;

  field_solver();
  virtual ~field_solver();

  virtual void update_fields(sim_data& data, double dt,
                             double time = 0.0) = 0;

 protected:
}; // ----- end of class field_solver -----


} // namespace Aperture

#endif  // _FIELD_SOLVER_H_
