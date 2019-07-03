#ifndef _PTC_UPDATER_1DGR_H_
#define _PTC_UPDATER_1DGR_H_

#include "grids/grid_1dgr.h"
#include "core/multi_array.h"

namespace Aperture {

struct sim_data;
class sim_environment;

class ptc_updater_1dgr {
 public:
  ptc_updater_1dgr(sim_environment& env);
  ~ptc_updater_1dgr();

  void update_particles(sim_data& data, double dt, uint32_t step = 0);
  void prepare_initial_condition(sim_data& data, int multiplicity);
  void prepare_initial_photons(sim_data& data, int multiplicity);

 protected:
  sim_environment& m_env;

  multi_array<Scalar> m_tmp_j;
};

}  // namespace Aperture



#endif  // _PTC_UPDATER_1DGR_H_
