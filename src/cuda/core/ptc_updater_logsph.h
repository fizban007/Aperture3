#ifndef _PTC_UPDATER_LOGSPH_H_
#define _PTC_UPDATER_LOGSPH_H_

#include "cuda/core/ptc_updater_dev.h"
#include "cuda/data/fields_dev.h"
#include "cuda/data/array.h"
#include "cuda/grids/grid_log_sph_dev.h"

namespace Aperture {

class PtcUpdaterLogSph : public PtcUpdaterDev {
 public:
  PtcUpdaterLogSph(const cu_sim_environment& env);
  virtual ~PtcUpdaterLogSph();

  virtual void update_particles(cu_sim_data& data, double dt,
                                uint32_t step = 0) override;
  virtual void handle_boundary(cu_sim_data& data) override;
  void inject_ptc(cu_sim_data& data, int inj_per_cell, Scalar p1,
                  Scalar p2, Scalar p3, Scalar w, Scalar omega);
  // void annihilate_extra_pairs(cu_sim_data& data);

 private:
  // Grid_LogSph_dev::mesh_ptrs m_mesh_ptrs;
  std::vector<void*> d_rand_states;
  int m_threadsPerBlock, m_blocksPerGrid;

  std::vector<cu_array<Scalar>> m_surface_e, m_surface_p;
  std::vector<cu_multi_array<Scalar>> m_tmp_j1, m_tmp_j2;
};  // ----- end of class PtcUpdaterLogSph : public PtcUpdaterDev -----

}  // namespace Aperture

#endif  // _PTC_UPDATER_LOGSPH_H_
