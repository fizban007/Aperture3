#ifndef _GRID_LOG_SPH_BASE_H_
#define _GRID_LOG_SPH_BASE_H_

#include "data/grid.h"
#include <cmath>

namespace Aperture {

template <typename Derived>
class Grid_LogSph_base : public Grid {
 public:
  Grid_LogSph_base() {}
  virtual ~Grid_LogSph_base() {}

  static void coord_to_cart(float& x, float& y, float& z,
                            const float& x1, const float& x2,
                            const float& x3) {
    float r = std::exp(x1);
    x = r * std::sin(x2);
    z = r * std::cos(x2);
  }

  virtual void init(const SimParams& params) override {}
};  // ----- end of class Grid_LogSph : public Grid -----

}  // namespace Aperture

#endif  // _GRID_LOG_SPH_BASE_H_
