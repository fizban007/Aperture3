#ifndef _RT_IC_H_
#define _RT_IC_H_

#include "cuda/cuda_control.h"
#include "cuda/data/cu_multi_array.h"
#include "cuda/data/array.h"
#include <random>

namespace Aperture {

struct SimParams;

class inverse_compton {
 public:
  inverse_compton(const SimParams& params);
  ~inverse_compton();

  template <typename F>
  void init(const F& n_e, Scalar emin, Scalar emax);

  cu_array<Scalar>& rate() { return m_rate; }
  cu_array<Scalar>& gammas() { return m_gammas; }
  cu_array<Scalar>& ep() { return m_ep; }
  // cu_multi_array<Scalar>& np() { return m_npep; }
  // cu_multi_array<Scalar>& dnde1p() { return m_dnde1p; }

  int find_n_gamma(Scalar gamma) const;
  Scalar find_n_ep(Scalar ep) const;
  Scalar gen_e1p(int gn);
  Scalar gen_ep(int gn, Scalar e1p);
  Scalar gen_photon_e(Scalar gamma);
  int binary_search(float u, int n, const cu_multi_array<Scalar>& array,
                    Scalar& v1, Scalar& v2) const;

  void generate_photon_energies(cu_array<Scalar>& e_ph, cu_array<Scalar>& gammas);
  void generate_random_gamma(cu_array<Scalar>& gammas);

 private:
  // cu_multi_array<Scalar> m_npep;
  // cu_multi_array<Scalar> m_dnde1p;
  cu_multi_array<Scalar> m_dNde;
  cu_array<Scalar> m_rate, m_gammas, m_ep;
  Scalar m_dep, m_dg;
  int m_threads, m_blocks;

  std::default_random_engine m_generator;
  std::uniform_real_distribution<float> m_dist;
  void* m_states;

};  // ----- end of class inverse_compton -----

}  // namespace Aperture

#endif  // _RT_IC_H_
