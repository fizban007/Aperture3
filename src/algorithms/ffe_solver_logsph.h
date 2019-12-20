#ifndef _FFE_SOLVER_LOGSPH_H_
#define _FFE_SOLVER_LOGSPH_H_

#include "core/fields.h"

namespace Aperture {

struct sim_data;
class sim_environment;

class ffe_solver_logsph {
 public:
  ffe_solver_logsph(sim_environment& env);
  ~ffe_solver_logsph();

  void update_fields(sim_data& data, double dt, double time = 0.0);
  void apply_boundary(sim_data& data, double omega, double time = 0.0);

 private:
  void copy_fields(const sim_data& data);
  void rk_push(sim_data& data, double dt);
  void rk_update(sim_data& data, Scalar c1, Scalar c2, Scalar c3);
  void clean_epar(sim_data& data);
  void check_eGTb(sim_data& data);

  sim_environment& m_env;

  vector_field<Scalar> En, dE;
  vector_field<Scalar> Bn, dB;
  scalar_field<Scalar> rho;
};

}  // namespace Aperture

#endif  // _FFE_SOLVER_LOGSPH_H_
