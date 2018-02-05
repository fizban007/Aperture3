#ifndef _SIM_DATA_H_
#define _SIM_DATA_H_

#include "data/enum_types.h"
#include "data/fields.h"
#include "data/grid.h"
#include "data/particles.h"
#include "data/photons.h"
#include "sim_environment.h"

namespace Aperture {

struct SimData {
  // SimData();
  SimData(const Environment& env);
  ~SimData();

  void initialize(const Environment& env);

  const Environment& env;
  VectorField<Scalar> E;
  VectorField<Scalar> B;
  ScalarField<Scalar> Bflux;
  VectorField<Scalar> J;
  std::vector<ScalarField<Scalar> > Rho;

  std::vector<Particles> particles;  // Each species occupies an array
  Photons photons;
  int num_species;
  double time = 0.0;
};
}

#endif  // _SIM_DATA_H_
