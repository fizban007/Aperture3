#ifndef _CU_SIM_DATA1D_H_
#define _CU_SIM_DATA1D_H_

#include "core/enum_types.h"
#include "cuda/core/cu_sim_environment.h"
#include "cuda/data/fields_dev.h"
#include "cuda/data/particles_1d.h"
#include "cuda/data/photons_1d.h"
#include <vector>

namespace Aperture {

struct cu_sim_data1d {
  cu_sim_data1d(const cu_sim_environment& env);
  virtual ~cu_sim_data1d();

  void initialize(const cu_sim_environment& env);
  // void prepare_initial_condition(int multiplicity);

  const cu_sim_environment& env;
  cu_vector_field<Scalar> E;
  // cu_vector_field<Scalar> B;
  cu_vector_field<Scalar> J;
  std::vector<cu_scalar_field<Scalar>> Rho;

  std::unique_ptr<Grid> grid;

  Particles_1D particles;
  Photons_1D photons;
  int dev_id;
  // int> dev_map;

  void init_grid(const cu_sim_environment& env);
  // void fill_multiplicity(Scalar weight, int multiplicity);
  void prepare_initial_condition(int multiplicity);
  // template <class Func>
  // void init_bg_B_field(int component, const Func& f);
  // template <class Func>
  // void init_bg_E_field(int component, const Func& f);
};  // ----- end of class cu_sim_data1d -----

}  // namespace Aperture

#endif  // _CU_SIM_DATA1D_H_
