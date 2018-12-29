#ifndef _PTC_UPDATER_H_
#define _PTC_UPDATER_H_

namespace Aperture {

struct sim_data;
class sim_environment;

class ptc_updater
{
 public:
  ptc_updater(const sim_environment& env) : m_env(env) {}
  virtual ~ptc_updater() {}

  virtual void update_particles(sim_data& data, double dt) = 0;
  virtual void handle_boundary(sim_data& data) = 0;

 protected:
  const sim_environment& m_env;
}; // ----- end of class ptc_updater -----


}

#endif  // _PTC_UPDATER_H_
