#ifndef _RADIATION_FIELD_H_
#define _RADIATION_FIELD_H_

#include "data/cu_multi_array.h"
#include "core/typedefs.h"

namespace Aperture {

class cu_sim_environment;

// This is tailored to handle radiation field in 1D
class RadiationField {
 public:
  RadiationField(const cu_sim_environment& env);
  virtual ~RadiationField();

  void advect(Scalar dt);

  cu_multi_array<Scalar>& data() { return m_data; }
  const cu_multi_array<Scalar>& data() { return m_data; }
  Scalar* ptr() { return m_data.data(); }
  const Scalar* ptr() { return m_data.data(); }

 private:
  const cu_sim_environment& m_env;
  cu_multi_array<Scalar> m_data;
};  // ----- end of class RadiationField -----

}  // namespace Aperture

#endif  // _RADIATION_FIELD_H_
