#ifndef _CU_SIM_DATA_H_
#define _CU_SIM_DATA_H_

#include "core/enum_types.h"
#include "core/grid.h"
#include "cuda/core/sim_environment_dev.h"
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
    for (unsigned int n = 0; n < dev_map.size(); n++) {
      cudaSetDevice(dev_map[n]);
      Bbg[n].initialize(component, f);
      Logger::print_debug("on host, B0 is {}", Bbg[n](0, 5, 4));
      // B[n].sync_to_device();
    }
  }
  template <class Func>
  void init_bg_E_field(int component, const Func& f) {
    for (unsigned int n = 0; n < dev_map.size(); n++) {
      cudaSetDevice(dev_map[n]);
      Ebg[n].initialize(component, f);
      // E[n].sync_to_device();
    }
  }

  void send_particles();
  void sort_particles();
  void init_bg_fields();
  void check_dev_mesh();
  void check_mesh_ptrs();

  const cu_sim_environment& env;
  std::vector<cu_vector_field<Scalar>> E;
  std::vector<cu_vector_field<Scalar>> B;
  std::vector<cu_vector_field<Scalar>> J;
  std::vector<cu_scalar_field<Scalar>> flux;
  std::vector<std::vector<cu_scalar_field<Scalar>>> Rho;
  std::vector<std::vector<cu_scalar_field<Scalar>>> gamma;
  std::vector<std::vector<cu_scalar_field<Scalar>>> ptc_num;
  std::vector<cu_scalar_field<Scalar>> divE;
  std::vector<cu_scalar_field<Scalar>> divB;
  std::vector<cu_scalar_field<Scalar>> photon_produced;
  std::vector<cu_scalar_field<Scalar>> pair_produced;
  std::vector<cu_scalar_field<Scalar>> photon_num;
  // std::vector<cu_scalar_field<Scalar>> Rho_avg;
  // std::vector<cu_scalar_field<Scalar>> J_s;
  // std::vector<cu_scalar_field<Scalar>> J_avg;

  std::vector<cu_vector_field<Scalar>> Ebg;
  std::vector<cu_vector_field<Scalar>> Bbg;

  std::vector<std::unique_ptr<Grid>> grid;

  std::vector<Particles> particles;
  std::vector<Photons> photons;

  std::vector<Particles> ptc_buffer;
  std::vector<Photons> ph_buffer;
  int num_species;
  double time = 0.0;
  // int devId;
  std::vector<int> dev_map;
};

}  // namespace Aperture

// #include "cu_sim_data_impl.hpp"

#endif  // _CU_SIM_DATA_H_
