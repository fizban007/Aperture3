#ifndef _CU_SIM_DATA_H_
#define _CU_SIM_DATA_H_

#include "core/enum_types.h"
#include "cuda/core/sim_environment_dev.h"
#include "cuda/data/fields_dev.h"
#include "cuda/data/particles_dev.h"
#include "cuda/data/photons_dev.h"

namespace Aperture {

struct cu_sim_data {
  // cu_sim_data();
  cu_sim_data(const cu_sim_environment& env, int deviceId = 0);
  ~cu_sim_data();

  void initialize(const cu_sim_environment& env);

  const cu_sim_environment& env;
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
  cu_multi_array<float> photon_flux;

  Particles particles;
  Photons photons;
  int num_species;
  double time = 0.0;
  int devId;

  void set_initial_condition(Scalar weight, int multiplicity);
};

}  // namespace Aperture

#endif  // _CU_SIM_DATA_H_
