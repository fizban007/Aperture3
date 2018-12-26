#ifndef _SIM_DATA_H_
#define _SIM_DATA_H_

#include "data/enum_types.h"
#include "data/fields.h"
#include <vector>

namespace Aperture {

class Environment;

struct sim_data
{
  sim_data(const Environment& env);
  ~sim_data();

  void initialize(const Environment& env);

  const Environment& env;
  vector_field<Scalar> E;
  vector_field<Scalar> B;
  vector_field<Scalar> J;
  std::vector<scalar_field<Scalar>> Rho;
  // std::vector<scalar_field<Scalar>> J_s;
  scalar_field<Scalar> flux;

  // particles_t particles;
  // photons_t photons;
  int num_species;
  double time = 0.0;
};

}

#endif  // _SIM_DATA_H_
