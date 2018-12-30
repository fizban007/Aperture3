#ifndef _PTC_UPDATER_DEFAULT_H_
#define _PTC_UPDATER_DEFAULT_H_

#include "ptc_updater.h"

namespace Aperture {

class ptc_updater_default : public ptc_updater
{
 public:
  ptc_updater_default(const sim_environment& env);
  virtual ~ptc_updater_default();

  virtual void update_particles(sim_data& data, double dt) override;

  // Reference implementation for benchmarking purposes
  void update_particles_slow(sim_data& data, double dt);
  virtual void handle_boundary(sim_data& data) override;

}; // ----- end of class ptc_updater_default : public ptc_updater -----


}

#endif  // _PTC_UPDATER_DEFAULT_H_
