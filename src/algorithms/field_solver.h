#ifndef _FIELD_SOLVER_H_
#define _FIELD_SOLVER_H_

namespace Aperture {

struct sim_data;
class sim_environment;

class field_solver {
 public:
  field_solver(sim_environment& env);
  ~field_solver();

  void update_fields(sim_data& data, double dt, double time = 0.0);
  void apply_outflow_boundary(sim_data& data, double time = 0.0);

 private:
  sim_environment& m_env;
};

}  // namespace Aperture

#endif  // _FIELD_SOLVER_H_
