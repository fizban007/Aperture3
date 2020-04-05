#ifndef _FIELD_SOLVER_LOGSPH_H_
#define _FIELD_SOLVER_LOGSPH_H_

#include "core/fields.h"
#include "grids/grid_log_sph.h"

namespace Aperture {

struct sim_data;
class sim_environment;

class field_solver_logsph {
 public:
  field_solver_logsph(sim_environment& env);
  ~field_solver_logsph();

  void update_fields(sim_data& data, double dt, double time = 0.0);
  void update_fields_semi_impl(sim_data& data, double alpha,
                               double beta, double dt,
                               double time = 0.0);
  void update_e_semi_impl(vector_field<Scalar>& e,
                          vector_field<Scalar>& j,
                          vector_field<Scalar>& b_old,
                          vector_field<Scalar>& b_new,
                          vector_field<Scalar>& b_bg,
                          double dt);
  void apply_boundary(sim_data& data, double omega, double time = 0.0);
  void filter_field(vector_field<Scalar>& field, int comp,
                    Grid_LogSph& grid);
  void compute_double_circ(vector_field<Scalar>& field,
                           vector_field<Scalar>& field_bg,
                           vector_field<Scalar>& result,
                           double coef);
  void compute_implicit_rhs(sim_data& data, double alpha, double beta,
                            vector_field<Scalar>& result,
                            double dt);
  void compute_divs(sim_data& data);

 private:
  sim_environment& m_env;

  multi_array<Scalar> m_tmp_e;
  vector_field<Scalar> m_tmp_b1, m_tmp_b2;
};

}  // namespace Aperture

#endif  // _FIELD_SOLVER_LOGSPH_H_
