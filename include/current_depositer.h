#ifndef _CURRENT_DEPOSITER_H_
#define _CURRENT_DEPOSITER_H_

#include "data/callbacks.h"
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
  void register_current_callback(const vfield_comm_callback& callback) {
    m_comm_J = callback;
  }
  void register_rho_callback(const sfield_comm_callback& callback) {
    m_comm_rho = callback;
  }

 protected:
  vfield_comm_callback m_comm_J;
  sfield_comm_callback m_comm_rho;
};  // ----- end of class current_depositer -----

}  // namespace Aperture

#endif  // _CURRENT_DEPOSITER_H_
