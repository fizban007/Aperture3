#ifndef _SIM_DATA_H_
#define _SIM_DATA_H_

#include "data/enum_types.h"
#include "data/fields.h"
#include "data/grid.h"
#include "data/particles.h"
// #include "data/photons.h"
#include "sim_environment.h"

namespace Aperture {

struct SimData {
  // SimData();
  SimData(const Environment& env, int deviceId = 0);
  ~SimData();

  void initialize(const Environment& env);

  const Environment& env;
  VectorField<Scalar> E;
  VectorField<Scalar> B;
  VectorField<Scalar> J;
  ScalarField<Scalar> flux;
  std::vector<ScalarField<Scalar>> Rho;
  std::vector<ScalarField<Scalar>> Rho_avg;
  std::vector<ScalarField<Scalar>> J_s;
  std::vector<ScalarField<Scalar>> J_avg;

  Particles particles;
  Photons photons;
  int num_species;
  double time = 0.0;
  int devId;
};

}  // namespace Aperture

#endif  // _SIM_DATA_H_
