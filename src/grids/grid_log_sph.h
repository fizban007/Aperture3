#ifndef _GRID_LOG_SPH_H_
#define _GRID_LOG_SPH_H_

#include "core/grid.h"
#include "core/multi_array.h"
#include "core/fields.h"

namespace Aperture {

class Grid_LogSph : public Grid {
 public:
  Grid_LogSph();
  virtual ~Grid_LogSph();

  virtual void compute_coef(const SimParams& params) override;

  static void coord_to_cart(float& x, float& y, float& z,
                            const float& x1, const float& x2,
                            const float& x3) {
    float r = std::exp(x1);
    x = r * std::sin(x2);
    z = r * std::cos(x2);
  }

  Scalar alpha(Scalar r, Scalar rs) const { return std::sqrt(1.0 - rs / r); }
  Scalar l1(Scalar r, Scalar rs) const {
    Scalar a = alpha(r, rs);
    return r * a + 0.5 * rs * std::log(2.0 * r * (1.0 + a) - rs);
  }
  Scalar A2(Scalar r, Scalar rs) const {
    Scalar a = alpha(r, rs);
    return 0.25 * r * a * (2.0 * r + 3.0 * rs) +
           0.375 * rs * rs * std::log(2.0 * r * (1.0 + a) - rs);
  }
  Scalar V3(Scalar r, Scalar rs) const {
    Scalar a = alpha(r, rs);
    return r * a * (8.0 * r * r + 10.0 * r * rs + 15.0 * rs * rs) / 24.0 +
        0.3125 * rs * rs * rs * std::log(2.0 * r * (1.0 + a) - rs);
  }

  void compute_flux(scalar_field<Scalar>& flux,
                    vector_field<Scalar>& B,
                    vector_field<Scalar>& B_bg) const;

  multi_array<Scalar> m_l1_e, m_l2_e, m_l3_e;
  multi_array<Scalar> m_l1_b, m_l2_b, m_l3_b;
  multi_array<Scalar> m_A1_e, m_A2_e, m_A3_e;
  multi_array<Scalar> m_A1_b, m_A2_b, m_A3_b;
  multi_array<Scalar> m_dV;
};  // ----- end of class Grid_LogSph : public
    // Grid_LogSph_base<Grid_LogSph> -----

}  // namespace Aperture

#endif  // _GRID_LOG_SPH_H_
