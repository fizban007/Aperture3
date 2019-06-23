#ifndef _PTC_UPDATER_1DGR_H_
#define _PTC_UPDATER_1DGR_H_

#include "grids/grid_1dgr.h"

namespace Aperture {

struct sim_data;
class sim_environment;

class ptc_updater_1dgr {
 public:
  ptc_updater_1dgr(sim_environment& env);
  ~ptc_updater_1dgr();

  void update_particles(sim_data& data, double dt, uint32_t step = 0);

 protected:
  sim_environment& m_env;
};

}  // namespace Aperture



#endif  // _PTC_UPDATER_1DGR_H_
