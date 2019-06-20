#ifndef _PTC_UPDATER_DEV_H_
#define _PTC_UPDATER_DEV_H_

#include "cuda/utils/pitchptr.cuh"
#include <vector>

namespace Aperture {

struct cu_sim_data;
class cu_sim_environment;

class PtcUpdaterDev {
 public:
  PtcUpdaterDev(const cu_sim_environment& env);
  virtual ~PtcUpdaterDev();

  virtual void update_particles(cu_sim_data& data, double dt,
                                uint32_t step = 0) = 0;
  virtual void handle_boundary(cu_sim_data& data) = 0;

  struct fields_data {
    pitchptr<Scalar> E1, E2, E3;
    pitchptr<Scalar> B1, B2, B3;
    pitchptr<Scalar> J1, J2, J3;
    pitchptr<Scalar>* Rho;
  };

 protected:
  void initialize_dev_fields(cu_sim_data& data);

  const cu_sim_environment& m_env;

  fields_data m_dev_fields;
  // std::vector<Extent> m_extent;
  bool m_fields_initialized;
};  // ----- end of class PtcUpdaterDev -----

}  // namespace Aperture

#endif  // _PTC_UPDATER_DEV_H_
