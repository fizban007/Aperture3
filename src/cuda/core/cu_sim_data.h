#ifndef _CU_SIM_DATA_H_
#define _CU_SIM_DATA_H_

#include "core/enum_types.h"
#include "core/grid.h"
#include "cuda/core/cu_sim_environment.h"
#include "cuda/data/fields_dev.h"
#include "cuda/data/detail/fields_dev_impl.hpp"
#include "cuda/data/particles_dev.h"
#include "cuda/data/photons_dev.h"
#include <vector>

namespace Aperture {

struct cu_sim_data {
  // cu_sim_data();
  cu_sim_data(const cu_sim_environment& env);
  ~cu_sim_data();

  void initialize(const cu_sim_environment& env);

  void init_grid(const cu_sim_environment& env);
  // void init_bg_fields();
  // void set_initial_condition(Scalar weight, int multiplicity);
  void fill_multiplicity(Scalar weight, int multiplicity);
  template <class Func>
  void init_bg_B_field(int component, const Func& f) {
    Bbg.initialize(component, f);
  }
  template <class Func>
  void init_bg_E_field(int component, const Func& f) {
    Ebg.initialize(component, f);
  }

  void send_particles();
  void sort_particles();
  void init_bg_fields();
  void check_dev_mesh();
  void check_mesh_ptrs();
  void compute_edotb();

  const cu_sim_environment& env;
  cu_vector_field<Scalar> E;
  cu_vector_field<Scalar> B;
  cu_vector_field<Scalar> J;
  std::vector<cu_scalar_field<Scalar>> Rho;
  std::vector<cu_scalar_field<Scalar>> gamma;
  std::vector<cu_scalar_field<Scalar>> ptc_num;
  cu_scalar_field<Scalar> flux;
  cu_scalar_field<Scalar> divE;
  cu_scalar_field<Scalar> divB;
  cu_scalar_field<Scalar> EdotB;
  cu_scalar_field<Scalar> photon_produced;
  cu_scalar_field<Scalar> pair_produced;
  cu_scalar_field<Scalar> photon_num;
  // std::vector<cu_scalar_field<Scalar>> Rho_avg;
  // std::vector<cu_scalar_field<Scalar>> J_s;
  // std::vector<cu_scalar_field<Scalar>> J_avg;

  cu_vector_field<Scalar> Ebg;
  cu_vector_field<Scalar> Bbg;

  std::unique_ptr<Grid> grid;

  Particles particles;
  Photons photons;

  std::vector<Particles> ptc_buffer;
  std::vector<Photons> ph_buffer;
  int num_species;
  double time = 0.0;
  // int devId;
  int dev_id;
};

}  // namespace Aperture

// #include "cu_sim_data_impl.hpp"

#endif  // _CU_SIM_DATA_H_
