#ifndef _PTC_UPDATER_1DGR_H_
#define _PTC_UPDATER_1DGR_H_

#include "cuda/grids/grid_1dgr_dev.h"
#include "cuda/utils/pitchptr.cuh"

namespace Aperture {

class cu_sim_environment;
struct cu_sim_data1d;

class ptc_updater_1dgr_dev {
 public:
  ptc_updater_1dgr_dev(const cu_sim_environment& env);
  virtual ~ptc_updater_1dgr_dev();

  void update_particles(cu_sim_data1d& data, double dt,
                        uint32_t step = 0);
  void initialize_dev_fields(cu_sim_data1d& data);

  struct fields_data {
    pitchptr<Scalar> E1, E3;
    // pitchptr<Scalar> B1, B3;
    pitchptr<Scalar> J1, J3;
    pitchptr<Scalar>* Rho;
  };

 private:
  const cu_sim_environment& m_env;
  // Grid_1dGR_dev::mesh_ptrs m_mesh_ptrs;

  bool m_fields_initialized = false;

  fields_data m_dev_fields;
};  // ----- end of class ptc_updater_1dgr_dev -----

}  // namespace Aperture

#endif  // _PTC_UPDATER_1DGR_H_
