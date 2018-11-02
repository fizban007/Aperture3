#ifndef _PTC_UPDATER_LOGSPH_H_
#define _PTC_UPDATER_LOGSPH_H_

#include "ptc_updater.h"
#include "data/grid_log_sph.h"

namespace Aperture {

class PtcUpdaterLogSph : public PtcUpdater {
 public:
  PtcUpdaterLogSph(const Environment& env);
  virtual ~PtcUpdaterLogSph();

  virtual void update_particles(SimData& data, double dt);
  virtual void handle_boundary(SimData& data);

 private:
  Grid_LogSph::mesh_ptrs m_mesh_ptrs;
};  // ----- end of class PtcUpdaterLogSph : public PtcUpdater -----

}  // namespace Aperture

#endif  // _PTC_UPDATER_LOGSPH_H_
