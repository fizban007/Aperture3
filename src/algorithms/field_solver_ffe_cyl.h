#ifndef _FIELD_SOLVER_FFE_CYL_H_
#define _FIELD_SOLVER_FFE_CYL_H_

#include "field_solver.h"

namespace Aperture {

class FieldSolver_FFE_Cyl : public FieldSolver {
 public:
  FieldSolver_FFE_Cyl(const Grid& g);
  virtual ~FieldSolver_FFE_Cyl();

  virtual void update_fields(SimData& data, double dt,
                             double time = 0.0) override;

  void compute_J(vfield_t& J, const vfield_t& E, const vfield_t& B);
  void update_field_substep(vfield_t& E_out, vfield_t& B_out,
                            vfield_t& J_out, const vfield_t& E_in,
                            const vfield_t& B_in, Scalar dt);
  virtual void set_background_j(const vfield_t& j) override {}

 private:
  void ffe_edotb(ScalarField<Scalar>& result,
                 const VectorField<Scalar>& E,
                 const VectorField<Scalar>& B, Scalar q = 1.0);
  void ffe_j(VectorField<Scalar>& result,
             const ScalarField<Scalar>& tmp_f,
             const VectorField<Scalar>& E, const VectorField<Scalar>& B,
             Scalar q = 1.0);
  void ffe_dE(VectorField<Scalar>& Eout, VectorField<Scalar>& J,
              const VectorField<Scalar>& E,
              const VectorField<Scalar>& B, Scalar dt);
  void ffe_reduceE(VectorField<Scalar>& E_center, const VectorField<Scalar>& E,
                   const VectorField<Scalar>& B);

  // sfield_t m_sf;
  vfield_t m_Etmp, m_Btmp;
};  // ----- end of class FieldSolver_FFE_Cyl -----

}  // namespace Aperture

#endif  // _FIELD_SOLVER_FFE_CYL_H_
