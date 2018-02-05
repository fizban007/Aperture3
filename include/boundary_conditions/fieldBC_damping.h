#ifndef _FIELDBC_DAMPING_H_
#define _FIELDBC_DAMPING_H_

#include "boundary_conditions/fieldBC.h"
#include <array>

namespace Aperture {

class fieldBC_damping : public fieldBC
{
 public:
  fieldBC_damping(BoundaryPos pos, int thickness, double coef, const Environment& env);
  virtual ~fieldBC_damping();

  virtual void initialize(const Environment& env, const SimData& data) override;
  virtual void apply (SimData& data, double time = 0) const override;
  virtual void apply (vfield_t& E, vfield_t& B, double time = 0) const override;
  virtual void apply (vfield_t& J, sfield_t& rho, double time = 0) const override;

 private:
  int m_thickness;
  double m_coef;
  // vfield_t m_B_bg;
  // vfield_t m_E_bg;
  // Grid m_grid_damp;
  Index m_grid_pos;
  std::array<MultiArray<Scalar>, 3> m_bg_E;
  std::array<MultiArray<Scalar>, 3> m_bg_B;
}; // ----- end of class fieldBC_rotating_conductor : public fieldBC -----


}

#endif  // _FIELDBC_DAMPING_H_
