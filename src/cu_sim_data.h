#ifndef _CU_SIM_DATA_H_
#define _CU_SIM_DATA_H_

#include "data/enum_types.h"
#include "data/fields_dev.h"
#include "data/grid.h"
#include "data/particles_dev.h"
#include "data/photons.h"
#include "sim_environment_dev.h"

namespace Aperture {

struct cu_sim_data {
  // cu_sim_data();
  cu_sim_data(const Environment& env, int deviceId = 0);
  ~cu_sim_data();

  void initialize(const Environment& env);

  const Environment& env;
  cu_vector_field<Scalar> E;
  cu_vector_field<Scalar> B;
  cu_vector_field<Scalar> J;
  cu_scalar_field<Scalar> flux;
  std::vector<cu_scalar_field<Scalar>> Rho;
  std::vector<cu_scalar_field<Scalar>> Rho_avg;
  std::vector<cu_scalar_field<Scalar>> J_s;
  std::vector<cu_scalar_field<Scalar>> J_avg;

  cu_vector_field<Scalar> Ebg;
  cu_vector_field<Scalar> Bbg;

  Particles particles;
  Photons photons;
  int num_species;
  double time = 0.0;
  int devId;
};

}  // namespace Aperture

#endif  // _CU_SIM_DATA_H_
