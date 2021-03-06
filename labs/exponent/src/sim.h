#ifndef _SIM_H_
#define _SIM_H_

#include "cuda/data/array.h"
#include "cuda/radiation/rt_ic.h"
#include "cuda/radiation/rt_tpp.h"

namespace Aperture {

class cu_sim_environment;

struct exponent_sim {
  exponent_sim(cu_sim_environment& env, bool gg = true, bool tpp = true);
  ~exponent_sim();

  template <typename T>
  void init_spectra(const T& spec, double n0);
  void push_particles(Scalar Eacc, Scalar dt);
  void add_new_particles(int num, Scalar E);
  void produce_photons(Scalar dt);
  void produce_pairs(Scalar dt);
  void compute_spectrum();
  void sort_photons();
  void prepare_initial_condition(int num_ptc, Scalar E);

  cu_sim_environment& m_env;
  inverse_compton m_ic;
  triplet_pairs m_tpp;

  size_t ptc_num, ph_num;

  cu_array<Scalar> ptc_E;
  cu_array<Scalar> ph_E;
  cu_array<Scalar> ph_path;
  cu_array<Scalar> ptc_spec;
  cu_array<Scalar> ph_spec;

  int m_threads_per_block, m_blocks_per_grid;
  cu_array<int> m_num_per_block;
  cu_array<int> m_cumnum_per_block;
  cu_array<int> m_pos_in_block;
  void* d_rand_states;

  bool do_gg, do_tpp;
};  // ----- end of class exponent_sim -----

}  // namespace Aperture

#endif  // _SIM_H_
