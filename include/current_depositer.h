#ifndef _CURRENT_DEPOSITER_H_
#define _CURRENT_DEPOSITER_H_

#include "sim_data.h"
#include "sim_environment.h"

namespace Aperture {

class CurrentDepositer {
 public:
  typedef VectorField<Scalar> vfield;
  typedef ScalarField<Scalar> sfield;

  CurrentDepositer() {}
  virtual ~CurrentDepositer() {}

  virtual void deposit(SimData& data, double dt) = 0;

  void set_periodic(bool p) { m_periodic = p; }
  void set_interp_order(int n) { m_interp = n; }
  // void register_current_callback(const vfield_comm_callback& callback) {
  //   m_comm_J = callback;
  // }
  // void register_rho_callback(const sfield_comm_callback& callback) {
  //   m_comm_rho = callback;
  // }

 protected:
  bool m_periodic = false;
  int m_interp = 1;
 //  vfield_comm_callback m_comm_J;
 //  sfield_comm_callback m_comm_rho;
};  // ----- end of class current_depositer -----

}  // namespace Aperture

#endif  // _CURRENT_DEPOSITER_H_
