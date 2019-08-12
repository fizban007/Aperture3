#ifndef _RT_IC_H_
#define _RT_IC_H_

#include "core/multi_array.h"
#include "core/array.h"
#include "cuda/cuda_control.h"
#include <random>

namespace Aperture {

struct SimParams;

class inverse_compton {
 public:
  inverse_compton(const SimParams& params);
  ~inverse_compton();

  template <typename F>
  void init(const F& n_e, Scalar emin, Scalar emax, double n0 = 1.0);

  array<Scalar>& ic_rate() { return m_ic_rate; }
  array<Scalar>& gg_rate() { return m_gg_rate; }
  array<Scalar>& gammas() { return m_gammas; }
  array<Scalar>& ep() { return m_ep; }

  int find_n_gamma(Scalar gamma) const;
  Scalar gen_e1p(int gn);
  Scalar gen_ep(int gn, Scalar e1p);
  Scalar gen_photon_e(Scalar gamma);
  int binary_search(float u, int n, const multi_array<Scalar>& array,
                    Scalar& v1, Scalar& v2) const;

  void generate_photon_energies(array<Scalar>& e_ph, array<Scalar>& gammas);
  void generate_random_gamma(array<Scalar>& gammas);

  multi_array<Scalar>& dNde_thomson() { return m_dNde_thomson; }
  array<Scalar>& log_ep() { return m_log_ep; }

 private:
  multi_array<Scalar> m_dNde;
  multi_array<Scalar> m_dNde_thomson;
  array<Scalar> m_ic_rate, m_gg_rate, m_gammas, m_log_ep, m_ep;
  Scalar m_dep, m_dg, m_dlep;
  int m_threads, m_blocks;

  std::default_random_engine m_generator;
  std::uniform_real_distribution<float> m_dist;
  void* m_states;

};  // ----- end of class inverse_compton -----

}  // namespace Aperture

#endif  // _RT_IC_H_
