#ifndef _FIELD_SOLVER_FORCE_FREE_H_
#define _FIELD_SOLVER_FORCE_FREE_H_

#include "field_solver.h"

namespace Aperture {

class FieldSolver_FFE : public FieldSolver
{
 public:
  FieldSolver_FFE(const Grid& g);
  virtual ~FieldSolver_FFE();
  // virtual void update_fields(vfield_t& E, vfield_t& B, const vfield_t& J, double dt, double time = 0.0) override;
  virtual void update_fields(SimData& data, double dt, double time = 0.0) override;

  void compute_J(vfield_t& J, const vfield_t& E, const vfield_t& B);
  void update_field_substep(vfield_t& E_out, vfield_t& B_out, vfield_t& J_out,
                            const vfield_t& E_in, const vfield_t& B_in, Scalar dt);
  virtual void set_background_j(const vfield_t& j) override {}

 private:
  void ffe_edotb(ScalarField<Scalar>& result, const VectorField<Scalar>& E,
                 const VectorField<Scalar>& B, Scalar q = 1.0);
  void ffe_j(VectorField<Scalar>& result, const ScalarField<Scalar>& tmp_f,
             const VectorField<Scalar>& E, const VectorField<Scalar>& B,
             Scalar q = 1.0);
  void ffe_dE(VectorField<Scalar>& Eout, VectorField<Scalar>& J,
              const VectorField<Scalar>& E, const VectorField<Scalar>& B,
              Scalar dt);

  // sfield_t m_sf;
  vfield_t m_Etmp, m_Btmp;
  // vfield_t m_e1, m_e2, m_e3, m_e4;
  // vfield_t m_b1, m_b2, m_b3, m_b4;
  // vfield_t m_j1, m_j2, m_j3, m_j4;
}; // ----- end of class FieldSolver_FFE -----


}

#endif  // _FIELD_SOLVER_FORCE_FREE_H_
