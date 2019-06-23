#ifndef _PTC_UPDATER_H_
#define _PTC_UPDATER_H_

namespace Aperture {

struct sim_data;
class sim_environment;

class ptc_updater {
 public:
  ptc_updater(sim_environment& env);
  ~ptc_updater();

  void update_particles(sim_data& data, double dt, uint32_t step = 0);
  void apply_boundary(sim_data& data, double dt, uint32_t step = 0);

 protected:
  sim_environment& m_env;
};

}  // namespace Aperture

#endif  // _PTC_UPDATER_H_
