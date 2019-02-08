#ifndef _RT_IC_H_
#define _RT_IC_H_

#include "cuda/cuda_control.h"
#include "cuda/data/cu_multi_array.h"
#include "cuda/data/array.h"

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
  cu_multi_array<Scalar>& np() { return m_npep; }
  cu_multi_array<Scalar>& dnde1p() { return m_dnde1p; }

 private:
  HOST_DEVICE double sigma_ic(Scalar x) const;
  HOST_DEVICE double x_ic(Scalar gamma, Scalar e, Scalar mu) const;
  HOST_DEVICE double beta(Scalar gamma) const;
  HOST_DEVICE double sigma_rest(Scalar ep, Scalar e1p) const;

  cu_multi_array<Scalar> m_npep;
  cu_multi_array<Scalar> m_dnde1p;
  cu_array<Scalar> m_rate, m_gammas, m_ep;
  double m_dep;
};  // ----- end of class inverse_compton -----

}  // namespace Aperture

#endif  // _RT_IC_H_
