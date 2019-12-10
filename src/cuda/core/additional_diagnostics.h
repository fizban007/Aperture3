#ifndef _ADDITIONAL_DIAGNOSTICS_H_
#define _ADDITIONAL_DIAGNOSTICS_H_

#include "cuda/data/fields_dev.h"
#include "cuda/utils/pitchptr.h"
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

 private:
  const cu_sim_environment& m_env;

  void init_pointers(cu_sim_data& data);

  pitchptr<Scalar>* m_dev_gamma;
  pitchptr<Scalar>* m_dev_ptc_num;
  bool m_initialized = false;
}; // ----- end of class AdditionalDiagnostics -----



}

#endif  // _ADDITIONAL_DIAGNOSTICS_H_
