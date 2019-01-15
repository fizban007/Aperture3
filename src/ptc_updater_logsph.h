#ifndef _PTC_UPDATER_LOGSPH_H_
#define _PTC_UPDATER_LOGSPH_H_

#include "data/grid_log_sph.h"
#include "data/fields_dev.h"
#include "ptc_updater_dev.h"

namespace Aperture {

class PtcUpdaterLogSph : public PtcUpdaterDev {
 public:
  PtcUpdaterLogSph(const Environment& env);
  virtual ~PtcUpdaterLogSph();

  virtual void update_particles(cu_sim_data& data, double dt, uint32_t step = 0);
  virtual void handle_boundary(cu_sim_data& data);
  void inject_ptc(cu_sim_data& data, int inj_per_cell, Scalar p1, Scalar p2,
                  Scalar p3, Scalar w, Scalar omega);

 private:
  Grid_LogSph::mesh_ptrs m_mesh_ptrs;
  void* d_rand_states;
  int m_threadsPerBlock, m_blocksPerGrid;

  cu_scalar_field<double> m_J1, m_J2;
};  // ----- end of class PtcUpdaterLogSph : public PtcUpdaterDev -----

}  // namespace Aperture

#endif  // _PTC_UPDATER_LOGSPH_H_
