#ifndef _CURRENT_DEPOSIT_ESIRKEPOV_H_
#define _CURRENT_DEPOSIT_ESIRKEPOV_H_

#include "current_depositer.h"

namespace Aperture {

class cu_sim_environment;

class CurrentDepositer_Esirkepov : public CurrentDepositer {
 public:
  typedef cu_vector_field<Scalar> vfield;
  typedef cu_scalar_field<Scalar> sfield;

  CurrentDepositer_Esirkepov(const cu_sim_environment& env);
  virtual ~CurrentDepositer_Esirkepov();

  virtual void deposit(cu_sim_data& data, double dt);
  void normalize_current(const vfield& I, vfield& J);
  void normalize_density(const sfield& Q, sfield& rho, sfield& V);
  void normalize_velocity(const sfield& rho, sfield& V);

 private:
  void compute_delta_rho(vfield& J, sfield& Rho, const Particles& part,
                         double dt);
  void compute_delta_rho(sfield& J, sfield& Rho, const Particles& part,
                         double dt);

  void scan_current(vfield& J);
  void scan_current(sfield& J);

  const cu_sim_environment& m_env;
  // int m_deposit_order = 3;
  // int m_deriv_order;
};  // ----- end of class current_depositer_Esirkepov : public
    // current_depositer

}  // namespace Aperture

#endif  // _CURRENT_DEPOSIT_ESIRKEPOV_H_
