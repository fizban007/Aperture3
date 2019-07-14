#ifndef _FIELD_SOLVER_1DGAP_H_
#define _FIELD_SOLVER_1DGAP_H_

namespace Aperture {

struct sim_data;
class sim_environment;

class field_solver_1dgap {
 public:
  field_solver_1dgap(sim_environment& env);
  ~field_solver_1dgap();

  void update_fields(sim_data& data, double dt, double time = 0.0);

 private:
  sim_environment& m_env;
};

}  // namespace Aperture



#endif  // _FIELD_SOLVER_1DGAP_H_
