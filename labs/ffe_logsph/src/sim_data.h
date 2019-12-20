#ifndef _SIM_DATA_H_
#define _SIM_DATA_H_

#include "core/enum_types.h"
#include "core/fields.h"
#include "core/particles.h"
#include "core/photons.h"
#include "utils/logger.h"
#include <vector>

namespace Aperture {

class sim_environment;

struct sim_data {
  sim_data(const sim_environment& env);
  ~sim_data();

  void initialize(const sim_environment& env);
  void finalize();

  void fill_multiplicity(Scalar weight, int multiplicity);

  template <class Func>
  void init_bg_B_field(int component, const Func& f) {
    Bbg.initialize(component, f);
  }

  void sort_particles();
  void init_bg_fields();
  void check_dev_mesh();
  void compute_edotb();
  void copy_to_host();

  const sim_environment& env;
  vector_field<Scalar> E;
  vector_field<Scalar> B;
  // vector_field<Scalar> Ebg;
  vector_field<Scalar> Bbg;
  vector_field<Scalar> J;
  // std::vector<scalar_field<Scalar>> Rho;
  // std::vector<scalar_field<Scalar>> gamma;
  // std::vector<scalar_field<Scalar>> ptc_num;
  // std::vector<scalar_field<Scalar>> J_s;
  scalar_field<Scalar> divE;
  scalar_field<Scalar> divB;
  // scalar_field<Scalar> EdotB;
  // scalar_field<Scalar> photon_produced;
  // scalar_field<Scalar> pair_produced;
  // scalar_field<Scalar> photon_num;
  // multi_array<float> ph_flux;

  // particles_t particles;
  // photons_t photons;
  // int num_species;
  double time = 0.0;
  int devId;
  // void* d_rand_states;
};

}  // namespace Aperture

#endif  // _SIM_DATA_H_
