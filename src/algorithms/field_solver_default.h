#ifndef _FIELD_SOLVER_DEFAULT_H_
#define _FIELD_SOLVER_DEFAULT_H_

#include "field_solver.h"

namespace Aperture {

class FieldSolver_Default : public FieldSolver
{
 public:
  FieldSolver_Default(const Grid& g);
  virtual ~FieldSolver_Default();
  virtual void update_fields(SimData& data, double dt, double time = 0.0) override;
  void update_fields(vfield_t& E, vfield_t& B, const vfield_t& J, double dt, double time = 0.0);
  // virtual void compute_flux(const vfield_t& f, sfield_t& flux) override;

  void compute_E_update(vfield_t& E, const vfield_t& B, const vfield_t& J, double dt);
  void compute_B_update(vfield_t& B, const vfield_t& E, double dt);

  virtual void set_background_j(const vfield_t& J) override;

 private:
  vfield_t m_dE, m_dB;
  vfield_t m_background_j;
}; // ----- end of class FieldSolver_Default : public FieldSolver -----


}

#endif  // _FIELD_SOLVER_DEFAULT_H_
