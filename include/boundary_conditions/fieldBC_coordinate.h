#ifndef _FIELDBC_COORDINATE_H_
#define _FIELDBC_COORDINATE_H_

#include "boundary_conditions/fieldBC.h"

namespace Aperture {

class fieldBC_coordinate : public fieldBC
{
 public:
  fieldBC_coordinate(BoundaryPos pos, const Grid& grid, MetricType type);
  virtual ~fieldBC_coordinate();

  virtual void initialize(const Environment& env, const SimData& data) override;
  virtual void apply (SimData& data, double time = 0) const override;
  virtual void apply (vfield_t& E, vfield_t& B, double time = 0) const override;
  virtual void apply (vfield_t& J, sfield_t& rho, double time = 0) const override;

 private:
  Extent m_ext;
  MetricType m_type;
  // Extent m_extB;

  void apply_current (VectorField<Scalar>& Jfield, double time) const;
  void apply_spherical_axis(const Extent& ext, BoundaryPos pos, vfield_t& E,
                            vfield_t& B, double time) const;
}; // ----- end of class fieldBC_coordinate : public fieldBC -----

}


#endif  // _FIELDBC_COORDINATE_H_
