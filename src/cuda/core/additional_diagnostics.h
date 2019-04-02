#ifndef _ADDITIONAL_DIAGNOSTICS_H_
#define _ADDITIONAL_DIAGNOSTICS_H_

#include "cuda/data/fields_dev.h"
#include <vector>

namespace Aperture {

struct cu_sim_data;
class cu_sim_environment;

class AdditionalDiagnostics
{
 public:
  AdditionalDiagnostics(const cu_sim_environment& env);
  ~AdditionalDiagnostics();

  void collect_diagnostics(cu_sim_data& data);

  // cu_scalar_field<Scalar>& get_ph_num() { return m_ph_num; }
  // cu_scalar_field<Scalar>& get_gamma(int n) { return m_gamma[n]; }
  // cu_scalar_field<Scalar>& get_ptc_num(int n) { return m_ptc_num[n]; }

 private:
  const cu_sim_environment& m_env;

  void init_pointers(cu_sim_data& data);
  // cu_scalar_field<Scalar> m_ph_num;
  // std::vector<cu_scalar_field<Scalar>> m_gamma;
  // std::vector<cu_scalar_field<Scalar>> m_ptc_num;

  std::vector<cudaPitchedPtr*> m_dev_gamma;
  std::vector<cudaPitchedPtr*> m_dev_ptc_num;
  bool m_initialized = false;
}; // ----- end of class AdditionalDiagnostics -----



}

#endif  // _ADDITIONAL_DIAGNOSTICS_H_
