#ifndef _RADIATION_FIELD_H_
#define _RADIATION_FIELD_H_

#include "data/multi_array_dev.h"
#include "data/typedefs.h"

namespace Aperture {

class Environment;

// This is tailored to handle radiation field in 1D
class RadiationField {
 public:
  RadiationField(const Environment& env);
  virtual ~RadiationField();

  void advect(Scalar dt);

  multi_array_dev<Scalar>& data() { return m_data; }
  const multi_array_dev<Scalar>& data() { return m_data; }
  Scalar* ptr() { return m_data.data(); }
  const Scalar* ptr() { return m_data.data(); }

 private:
  const Environment& m_env;
  multi_array_dev<Scalar> m_data;
};  // ----- end of class RadiationField -----

}  // namespace Aperture

#endif  // _RADIATION_FIELD_H_
