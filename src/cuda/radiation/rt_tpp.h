#ifndef _RT_TPP_H_
#define _RT_TPP_H_

#include "cuda/cuda_control.h"
#include "cuda/data/cu_multi_array.h"
#include "cuda/data/array.h"
#include <random>

namespace Aperture {

class triplet_pairs
{
 public:
  triplet_pairs(const SimParams& params);
  ~triplet_pairs();

  template <typename F>
  void init(const F& n_gamma, Scalar emin, Scalar emax, double n0 = 1.0);

  cu_array<Scalar>& rate() { return m_rate; }
  cu_array<Scalar>& Em() { return m_Em; }

 private:
  cu_array<Scalar> m_rate, m_Em, m_gammas;
  cu_multi_array<Scalar> m_dNde;

  Scalar m_dg;
  int m_threads, m_blocks;

  std::default_random_engine m_generator;
  std::uniform_real_distribution<float> m_dist;
  void* m_states;
}; // ----- end of class triplet_pairs -----


}

#endif  // _RT_TPP_H_
