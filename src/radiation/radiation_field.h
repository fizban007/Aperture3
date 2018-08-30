#ifndef _RADIATION_FIELD_H_
#define _RADIATION_FIELD_H_

#include "data/typedefs.h"
#include "data/multi_array.h"

namespace Aperture {

class Environment;

// This is tailored to handle radiation field in 1D
class RadiationField
{
 public:
  RadiationField(const Environment& env);
  virtual ~RadiationField();

  void advect(Scalar dt);

  MultiArray<Scalar>& data() { return m_data; }
  const MultiArray<Scalar>& data() { return m_data; }
  Scalar* ptr() { return m_data.data(); }
  const Scalar* ptr() { return m_data.data(); }


 private:
  const Environment& m_env;
  MultiArray<Scalar> m_data;
}; // ----- end of class RadiationField -----


}

#endif  // _RADIATION_FIELD_H_
