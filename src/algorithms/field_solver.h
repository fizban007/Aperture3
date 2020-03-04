#ifndef _FIELD_SOLVER_H_
#define _FIELD_SOLVER_H_

#include <cstdint>

namespace Aperture {

struct sim_data;
class sim_environment;

class field_solver {
 public:
  field_solver(sim_environment& env);
  ~field_solver();

  void update_fields(sim_data& data, double dt, uint32_t step = 0);
  void apply_outflow_boundary(sim_data& data, double time = 0.0);
  void compute_divs(sim_data& data);

  void update_e_field(sim_data& data, double dt);
  void update_b_field(sim_data& data, double dt);

 private:
  sim_environment& m_env;
};

}  // namespace Aperture

#endif  // _FIELD_SOLVER_H_
