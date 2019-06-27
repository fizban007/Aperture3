#ifndef _FIELD_SOLVER_FFE_H_
#define _FIELD_SOLVER_FFE_H_

#include "core/fields.h"

namespace Aperture {

struct sim_data;
class sim_environment;

class field_solver_ffe {
 public:
  typedef vector_field<Scalar> vfield_t;

  field_solver_ffe(sim_environment& env);
  virtual ~field_solver_ffe();

  void update_fields(sim_data& data, double dt, double time = 0.0);

  void compute_J(vfield_t& J, vfield_t& E, vfield_t& B);
  void update_field_substep(vfield_t& E_out, vfield_t& B_out,
                            vfield_t& J_out, vfield_t& E_in,
                            vfield_t& B_in, Scalar dt);
  void apply_boundary(sim_data& data, double omega, double time = 0.0);

 private:
  void ffe_edotb(scalar_field<Scalar>& result,
                 vector_field<Scalar>& E,
                 vector_field<Scalar>& B, Scalar q = 1.0);
  void ffe_j(vector_field<Scalar>& result,
             scalar_field<Scalar>& tmp_f,
             vector_field<Scalar>& E,
             vector_field<Scalar>& B, Scalar q = 1.0);
  void ffe_dE(vector_field<Scalar>& Eout, vector_field<Scalar>& J,
              vector_field<Scalar>& E,
              vector_field<Scalar>& B, Scalar dt);

  // sfield_t m_sf;
  sim_environment& m_env;
  vfield_t m_Etmp, m_Btmp;
  // vfield_t m_e1, m_e2, m_e3, m_e4;
  // vfield_t m_b1, m_b2, m_b3, m_b4;
  // vfield_t m_j1, m_j2, m_j3, m_j4;
};  // ----- end of class FieldSolver_FFE -----

}  // namespace Aperture

#endif  // _FIELD_SOLVER_FFE_H_
