#ifndef _PTC_UPDATER_LOGSPH_H_
#define _PTC_UPDATER_LOGSPH_H_

#include "core/typedefs.h"
#include "core/array.h"
#include "core/multi_array.h"

namespace Aperture {

struct sim_data;
class sim_environment;

class ptc_updater_logsph {
 public:
  ptc_updater_logsph(sim_environment& env);
  ~ptc_updater_logsph();

  void update_particles(sim_data& data, double dt, uint32_t step = 0);
  void apply_boundary(sim_data& data, double dt, uint32_t step = 0);
  void inject_ptc(sim_data& data, int inj_per_cell, Scalar p1, Scalar p2, Scalar p3,
                  Scalar w, Scalar omega);

 protected:
  sim_environment& m_env;

  array<Scalar> m_surface_e, m_surface_p, m_surface_tmp;
  multi_array<Scalar> m_tmp_j1, m_tmp_j2;
};

}  // namespace Aperture



#endif  // _PTC_UPDATER_LOGSPH_H_
