#ifndef _PTC_UPDATER_1D_H_
#define _PTC_UPDATER_1D_H_

#include "ptc_updater.h"

namespace Aperture {

struct sim_data;
class sim_environment;

class ptc_updater_1d : public ptc_updater {
 public:
  ptc_updater_1d(const sim_environment& env);
  virtual ~ptc_updater_1d();

  virtual void update_particles(sim_data& data, double dt, uint32_t step = 0);
  virtual void handle_boundary(sim_data& data);

  void push(sim_data& data, double dt, uint32_t step);
  void deposit(sim_data& data, double dt, uint32_t step);
};  // ----- end of class ptc_updater_1d : public ptc_updater -----

}  // namespace Aperture

#endif  // _PTC_UPDATER_1D_H_
