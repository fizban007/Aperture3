#ifndef _PTC_UPDATER_DEFAULT_H_
#define _PTC_UPDATER_DEFAULT_H_

#include "ptc_updater.h"
#include "data/multi_array.h"
#include "utils/simd.h"

namespace Aperture {

class ptc_updater_default : public ptc_updater {
 public:
  ptc_updater_default(const sim_environment& env);
  virtual ~ptc_updater_default();

  virtual void update_particles(sim_data& data, double dt) override;

  // Reference implementation for benchmarking purposes
  void update_particles_slow(sim_data& data, double dt);

  // void vay_push(sim_data& data, double dt);
  // void esirkepov_deposit(sim_data& data, double dt);
  virtual void handle_boundary(sim_data& data) override;

 private:
  multi_array<simd::simd_buffer> m_j1;
  multi_array<simd::simd_buffer> m_j2;
  multi_array<simd::simd_buffer> m_j3;
};  // ----- end of class ptc_updater_default : public ptc_updater -----

}  // namespace Aperture

#endif  // _PTC_UPDATER_DEFAULT_H_
