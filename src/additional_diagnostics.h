#ifndef _ADDITIONAL_DIAGNOSTICS_H_
#define _ADDITIONAL_DIAGNOSTICS_H_

#include "data/fields.h"
#include <vector>

namespace Aperture {

struct SimData;
class Environment;

class AdditionalDiagnostics
{
 public:
  AdditionalDiagnostics(const Environment& env);
  ~AdditionalDiagnostics();

  void collect_diagnostics(const SimData& data);

  ScalarField<Scalar>& get_ph_num() { return m_ph_num; }
  ScalarField<Scalar>& get_gamma(int n) { return m_gamma[n]; }
  ScalarField<Scalar>& get_ptc_num(int n) { return m_ptc_num[n]; }

 private:
  const Environment& m_env;

  ScalarField<Scalar> m_ph_num;
  std::vector<ScalarField<Scalar>> m_gamma;
  std::vector<ScalarField<Scalar>> m_ptc_num;

  cudaPitchedPtr* m_dev_gamma;
  cudaPitchedPtr* m_dev_ptc_num;
}; // ----- end of class AdditionalDiagnostics -----



}

#endif  // _ADDITIONAL_DIAGNOSTICS_H_
