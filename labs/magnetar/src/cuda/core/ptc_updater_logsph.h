#ifndef _PTC_UPDATER_LOGSPH_H_
#define _PTC_UPDATER_LOGSPH_H_

#include "cuda/core/ptc_updater_dev.h"
#include "cuda/data/fields_dev.h"
#include "cuda/grids/grid_log_sph_dev.h"

namespace Aperture {

class PtcUpdaterLogSph : public PtcUpdaterDev {
 public:
  PtcUpdaterLogSph(const cu_sim_environment& env);
  virtual ~PtcUpdaterLogSph();

  virtual void update_particles(cu_sim_data& data, double dt,
                                uint32_t step = 0);
  virtual void handle_boundary(cu_sim_data& data);
  void inject_ptc(cu_sim_data& data, int inj_per_cell, Scalar p1,
                  Scalar p2, Scalar p3, Scalar w, Scalar omega);
  void annihilate_extra_pairs(cu_sim_data& data, double dt);

 private:
  Grid_LogSph_dev::mesh_ptrs m_mesh_ptrs;
  void* d_rand_states;
  int m_threadsPerBlock, m_blocksPerGrid;

  cu_scalar_field<Scalar> m_dens_e, m_dens_p, m_balance;
  cu_multi_array<int> m_annihilate;
};  // ----- end of class PtcUpdaterLogSph : public PtcUpdaterDev -----

}  // namespace Aperture

#endif  // _PTC_UPDATER_LOGSPH_H_
