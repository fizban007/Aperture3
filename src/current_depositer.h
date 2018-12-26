#ifndef _CURRENT_DEPOSITER_H_
#define _CURRENT_DEPOSITER_H_

#include "data/callbacks.h"
#include "sim_data_dev.h"

namespace Aperture {

class CurrentDepositer {
 public:
  CurrentDepositer() {}
  virtual ~CurrentDepositer() {}

  virtual void deposit(SimData& data, double dt) = 0;

  // void set_interp_order(int n) { m_interp = n; }
  void register_current_callback(const vfield_comm_callback& callback) {
    m_comm_J = callback;
  }
  void register_rho_callback(const sfield_comm_callback& callback) {
    m_comm_rho = callback;
  }

 protected:
  // int m_interp = 1;
  vfield_comm_callback m_comm_J;
  sfield_comm_callback m_comm_rho;
};  // ----- end of class current_depositer -----

}  // namespace Aperture

#endif  // _CURRENT_DEPOSITER_H_
