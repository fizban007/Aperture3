#ifndef _SIM_DATA_H_
#define _SIM_DATA_H_

#include "core/enum_types.h"
#include "core/fields.h"
#include "core/particles.h"
#include "core/photons.h"
#include "utils/logger.h"
#include "sim_environment.h"
#include <vector>

namespace Aperture {

struct sim_data {
  sim_data(sim_environment& env);
  ~sim_data();

  void initialize(sim_environment& env);
  void finalize();

  void fill_multiplicity(Scalar weight, int multiplicity);

  template <class Func>
  void init_bg_B_field(int component, const Func& f) {
    Bbg.initialize(component, f);
    B.initialize(component, f);
    env.send_array_guard_cells(Bbg.data(component));
    env.send_array_guard_cells(B.data(component));
  }
  template <class Func>
  void init_bg_E_field(int component, const Func& f) {
    Ebg.initialize(component, f);
    E.initialize(component, f);
    env.send_array_guard_cells(Ebg.data(component));
    env.send_array_guard_cells(E.data(component));
  }

  void sort_particles();
  void init_bg_fields();
  void check_dev_mesh();
  void compute_edotb();
  void copy_to_host();
  void copy_to_device();

  sim_environment& env;
  vector_field<Scalar> E;
  vector_field<Scalar> B;
  vector_field<Scalar> Ebg;
  vector_field<Scalar> Bbg;
  vector_field<Scalar> J;
  std::vector<scalar_field<Scalar>> Rho;
  std::vector<scalar_field<Scalar>> gamma;
  std::vector<scalar_field<Scalar>> ptc_num;
  // std::vector<scalar_field<Scalar>> J_s;
  scalar_field<Scalar> divE;
  scalar_field<Scalar> divB;
  scalar_field<Scalar> EdotB;
  scalar_field<Scalar> photon_produced;
  scalar_field<Scalar> pair_produced;
  scalar_field<Scalar> photon_num;
  multi_array<float> ph_flux;

  particles_t particles;
  photons_t photons;
  int num_species;
  double time = 0.0;
  int devId;
  void* d_rand_states;
  int rand_state_size = 0;
};

}  // namespace Aperture

#endif  // _SIM_DATA_H_
