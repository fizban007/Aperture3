#ifndef _PIC_SIM_H_
#define _PIC_SIM_H_

#include "current_depositer.h"
#include "field_solver.h"
#include "particle_pusher.h"
#include "sim_data.h"
#include "sim_environment.h"
#include <memory>
#include <stddef.h>

namespace Aperture {

/// Simulator class for PIC, bundling together different modules
class PICSim {
 public:
  // PICSim();
  PICSim(Environment& env);
  virtual ~PICSim();

  void loop(SimData& data, uint32_t steps = 100000,
            uint32_t data_interval = 100);

  void step(SimData& data, uint32_t step);

  FieldSolver& field_solver() { return *m_field_solver; }
  // InverseCompton& inverse_compton() { return *m_inverse_compton; }

 private:
  Environment& m_env;

  // modules
  std::unique_ptr<ParticlePusher> m_pusher;
  std::unique_ptr<CurrentDepositer> m_depositer;
  std::unique_ptr<FieldSolver> m_field_solver;
  // std::unique_ptr<InverseCompton> m_inverse_compton;
  // std::unique_ptr<DomainCommunicator> m_comm;
};  // ----- end of class PICSim -----
}  // namespace Aperture

#endif  // _PIC_SIM_H_
