#ifndef _FIELDBC_H_
#define _FIELDBC_H_

#include <functional>
#include "constant_defs.h"
#include "data/enum_types.h"
#include "data/fields.h"
#include "data/multi_array.h"
// #include "sim_data.h"
// #include "sim_environment.h"

namespace Aperture {

struct SimData;
class Environment;

class fieldBC {
 public:
  typedef MultiArray<Scalar> array_type;
  typedef VectorField<Scalar> vfield_t;
  typedef ScalarField<Scalar> sfield_t;

  fieldBC(BoundaryPos pos) : m_pos(pos) {}
  virtual ~fieldBC() {}

  virtual void initialize(const Environment& env, const SimData& data) = 0;
  // virtual void initialize(const initial_condition& ic, const Grid& grid) = 0;
  // virtual void initialize (const initial_condition& ic, const Grid& grid,
  // const Grid& grid_dual) = 0;
  virtual void apply(SimData& data, double time = 0) const = 0;
  virtual void apply(vfield_t& E, vfield_t& B, double time = 0) const = 0;
  virtual void apply(vfield_t& J, sfield_t& rho, double time = 0) const = 0;

  // virtual int thickness () = 0;

  BoundaryPos pos() { return m_pos; }

 protected:
  BoundaryPos m_pos;
};  // ----- end of class fieldBC -----

}  // namespace Aperture

#endif  // _FIELDBC_H_
