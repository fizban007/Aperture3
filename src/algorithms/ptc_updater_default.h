#ifndef _PTC_UPDATER_DEFAULT_H_
#define _PTC_UPDATER_DEFAULT_H_

#include "ptc_updater.h"

namespace Aperture {

class ptc_updater_default : public ptc_updater
{
 public:
  ptc_updater_default(const Environment& env);
  virtual ~ptc_updater_default();

  virtual void update_particles(sim_data& data, double dt) override;
  virtual void handle_boundary(sim_data& data) override;

}; // ----- end of class ptc_updater_default : public ptc_updater -----


}

#endif  // _PTC_UPDATER_DEFAULT_H_
