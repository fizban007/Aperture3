#ifndef _PTC_UPDATER_CART_H_
#define _PTC_UPDATER_CART_H_

#include "core/array.h"
#include "core/fields.h"
#include "core/multi_array.h"
#include "core/typedefs.h"

namespace Aperture {

struct sim_data;
class sim_environment;

class ptc_updater_cart {
 public:
  ptc_updater_cart(sim_environment& env);
  ~ptc_updater_cart();

  void update_particles(sim_data& data, double dt, uint32_t step = 0);
  void apply_boundary(sim_data& data, double dt, uint32_t step = 0);
  // void inject_ptc(sim_data& data, int inj_per_cell, Scalar p1, Scalar
  // p2, Scalar p3,
  //                 Scalar w, Scalar omega);

 protected:
  sim_environment& m_env;

  multi_array<Scalar> m_tmp_j1, m_tmp_j2;
  vector_field<Scalar> m_tmp_E, m_tmp_B;
};

}  // namespace Aperture

#endif  // _PTC_UPDATER_CART_H_
