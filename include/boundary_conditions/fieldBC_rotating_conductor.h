#ifndef _FIELDBC_ROTATING_CONDUCTOR_H_
#define _FIELDBC_ROTATING_CONDUCTOR_H_

#include "boundary_conditions/fieldBC.h"

namespace Aperture {

class fieldBC_rotating_conductor : public fieldBC
{
 public:
  fieldBC_rotating_conductor(BoundaryPos pos, int thickness, const Environment& env);
  virtual ~fieldBC_rotating_conductor();

  template <typename Func>
  auto set_time_dependence(const Func& f) -> decltype(*this);
  template <typename Func>
  auto set_spatial_dependence(const Func& f) -> decltype(*this);
  auto set_omega(double omega) -> decltype(*this);

 private:
  double m_omega;
}; // ----- end of class fieldBC_rotating_conductor : public fieldBC -----


}

#endif  // _FIELDBC_ROTATING_CONDUCTOR_H_
