#ifndef _PTC_UPDATER_LOGSPH_H_
#define _PTC_UPDATER_LOGSPH_H_

#include "data/grid_log_sph.h"
#include "data/fields_dev.h"
#include "ptc_updater.h"

namespace Aperture {

class PtcUpdaterLogSph : public PtcUpdater {
 public:
  PtcUpdaterLogSph(const Environment& env);
  virtual ~PtcUpdaterLogSph();

  virtual void update_particles(SimData& data, double dt);
  virtual void handle_boundary(SimData& data);
  void inject_ptc(SimData& data, int inj_per_cell, Scalar p1, Scalar p2,
                  Scalar p3, Scalar w, Scalar omega);

 private:
  Grid_LogSph::mesh_ptrs m_mesh_ptrs;
  void* d_rand_states;
  int m_threadsPerBlock, m_blocksPerGrid;

  ScalarField<double> m_J1, m_J2;
};  // ----- end of class PtcUpdaterLogSph : public PtcUpdater -----

}  // namespace Aperture

#endif  // _PTC_UPDATER_LOGSPH_H_
