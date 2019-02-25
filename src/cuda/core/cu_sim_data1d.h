#ifndef _CU_SIM_DATA1D_H_
#define _CU_SIM_DATA1D_H_

#include "core/enum_types.h"
#include "cuda/core/sim_environment_dev.h"
#include "cuda/data/fields_dev.h"
#include "cuda/data/particles_1d.h"
#include "cuda/data/photons_1d.h"

namespace Aperture {

struct cu_sim_data1d {
  cu_sim_data1d(const cu_sim_environment& env, int deviceId = 0);
  virtual ~cu_sim_data1d();

  void initialize(const cu_sim_environment& env);
  void prepare_initial_condition(int multiplicity);

  const cu_sim_environment& env;
  cu_vector_field<Scalar> E;
  // cu_vector_field<Scalar> B;
  cu_vector_field<Scalar> J;
  std::vector<cu_scalar_field<Scalar>> Rho;

  Particles_1D particles;
  Photons_1D photons;
  int devId;
};  // ----- end of class cu_sim_data1d -----

}  // namespace Aperture

#endif  // _CU_SIM_DATA1D_H_
