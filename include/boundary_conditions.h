#ifndef _BOUNDARY_CONDITIONS_H_
#define _BOUNDARY_CONDITIONS_H_

#include <array>
#include <memory>
#include "data/vec3.h"
#include "boundary_conditions/fieldBC.h"
#include "boundary_conditions/ptcBC.h"
// #include "sim_data.h"

namespace Aperture {

struct SimData;

class BoundaryConditions
{
 public:
  typedef VectorField<Scalar> vfield_t;
  typedef ScalarField<Scalar> sfield_t;

  BoundaryConditions();
  virtual ~BoundaryConditions();

  void add_fieldBC(fieldBC* bc);
  void add_ptcBC(ptcBC* bc);

  void apply_fieldBC(SimData& data) const;
  void apply_fieldBC(vfield_t& E, vfield_t& B, double time = 0.0) const;
  void apply_fieldBC(vfield_t& J, sfield_t& rho, double time = 0.0) const;

  void apply_ptcBC(SimData& data) const;

  bool* is_boundary() {
    return m_is_boundary.data();
  }
  void initialize(const Environment& env, const SimData& data);

  Index bound_start();
  Extent bound_ext();

 private:
  std::array<std::unique_ptr<fieldBC>, 6> m_field_bc;
  std::array<std::unique_ptr<ptcBC>, 6> m_ptc_bc;
  std::array<bool, 6> m_is_boundary;

}; // ----- end of class boundary_conditions -----

}

#endif  // _BOUNDARY_CONDITIONS_H_

#include "boundary_conditions/fieldBC_conductor.h"
#include "boundary_conditions/fieldBC_damping.h"
