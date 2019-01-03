#ifndef _FIELD_SOLVER_INTEGRAL_H_
#define _FIELD_SOLVER_INTEGRAL_H_

#include "field_solver_dev.h"
#include <vector>

namespace Aperture {

class FieldSolver_Integral : public FieldSolverDev {
 public:
  FieldSolver_Integral(const Grid& g, const Grid& g_dual);
  virtual ~FieldSolver_Integral();

  virtual void update_fields(vfield_t& E, vfield_t& B,
                             const vfield_t& J, double dt,
                             double time = 0.0) override;
  virtual void update_fields(cu_sim_data& data, double dt,
                             double time = 0.0) override;
  // virtual void compute_flux(const vfield_t& f, sfield_t& flux)
  // override;

  void compute_E_update(vfield_t& E, const vfield_t& B,
                        const vfield_t& J, double dt);
  void compute_B_update(vfield_t& B, const vfield_t& E, double dt);

  virtual void set_background_j(const vfield_t& J);

 private:
  vfield_t m_dE, m_dB;
  vfield_t m_background_j;
};  // ----- end of class field_solver_integral : public field_solver
    // -----

}  // namespace Aperture

#endif  // _FIELD_SOLVER_INTEGRAL_H_
