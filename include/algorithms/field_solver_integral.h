#ifndef _FIELD_SOLVER_INTEGRAL_H_
#define _FIELD_SOLVER_INTEGRAL_H_

#include <vector>
#include "field_solver.h"

namespace Aperture {

class FieldSolver_Integral : public FieldSolver
{
 public:
  FieldSolver_Integral(const Grid& g, const Grid& g_dual);
  virtual ~FieldSolver_Integral();

  virtual void update_fields(vfield_t& E, vfield_t& B, const vfield_t& J, double dt, double time = 0.0) override;
  virtual void update_fields(SimData& data, double dt, double time = 0.0) override;
  // virtual void compute_flux(const vfield_t& f, sfield_t& flux) override;

  void compute_E_update(vfield_t& E, const vfield_t& B, const vfield_t& J, double dt);
  void compute_B_update(vfield_t& B, const vfield_t& E, double dt);

  void compute_E_update_KS(vfield_t &E, const vfield_t &B, const vfield_t &I, double dt);
  void compute_B_update_KS(vfield_t &B, const vfield_t &E, double dt);

  void compute_auxiliary(const vfield_t& E, const vfield_t& B);
  void compute_GR_auxiliary(const vfield_t& E, const vfield_t& B);

 private:
  vfield_t m_dE, m_dB;
  MultiArray<Scalar> m_ks_rhs;
  std::vector<Scalar> m_tri_a, m_tri_c, m_tri_d;
  vfield_t m_E_aux, m_H_aux;
}; // ----- end of class field_solver_integral : public field_solver -----


}

#endif  // _FIELD_SOLVER_INTEGRAL_H_
