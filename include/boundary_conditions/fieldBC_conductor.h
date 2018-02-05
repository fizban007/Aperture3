#ifndef _FIELDBC_CONDUCTOR_H_
#define _FIELDBC_CONDUCTOR_H_

#include "boundary_conditions/fieldBC.h"

namespace Aperture {

class fieldBC_conductor : public fieldBC
{
 public:
  fieldBC_conductor(BoundaryPos pos, int thickness, const Environment& env);
  virtual ~fieldBC_conductor();

  virtual void initialize(const Environment& env, const SimData& data);
  virtual void apply (SimData& data, double time = 0) const;
  virtual void apply (vfield_t& E, vfield_t& B, double time = 0) const;
  virtual void apply (vfield_t& J, sfield_t& rho, double time = 0) const;

 private:
  int m_thickness;
  std::array<MultiArray<Scalar>, 3> m_boundary_E;
  std::array<MultiArray<Scalar>, 3> m_boundary_B;
}; // ----- end of class fieldBC_rotating_conductor : public fieldBC -----


}

#endif  // _FIELDBC_CONDUCTOR_H_
