#ifndef _FIELD_SOLVER_FFE_CYL_H_
#define _FIELD_SOLVER_FFE_CYL_H_

#include "field_solver_dev.h"

namespace Aperture {

class FieldSolver_FFE_Cyl : public FieldSolverDev {
 public:
  FieldSolver_FFE_Cyl(const Grid& g);
  virtual ~FieldSolver_FFE_Cyl();

  virtual void update_fields(SimData& data, double dt,
                             double omega = 0.0) override;

  void compute_J(vfield_t& J, const vfield_t& E, const vfield_t& B);
  void update_field_substep(vfield_t& E_out, vfield_t& B_out,
                            vfield_t& J_out, const vfield_t& E_in,
                            const vfield_t& B_in, Scalar omega, Scalar dt);
  void handle_boundary(SimData& data, Scalar omega, Scalar dt);
  virtual void set_background_j(const vfield_t& j) override {}

 private:
  void ffe_dE(cu_vector_field<Scalar>& Eout, cu_vector_field<Scalar>& J,
              const cu_vector_field<Scalar>& E,
              const cu_vector_field<Scalar>& B, Scalar dt);
  void ffe_reduceE(cu_vector_field<Scalar>& E_center, const cu_vector_field<Scalar>& E,
                   const cu_vector_field<Scalar>& B);

  // sfield_t m_sf;
  vfield_t m_Etmp, m_Etmp2;
  vfield_t m_Erk, m_Brk;

  Scalar m_a[4], m_b[4], m_c[4];
};  // ----- end of class FieldSolver_FFE_Cyl -----

}  // namespace Aperture

#endif  // _FIELD_SOLVER_FFE_CYL_H_
