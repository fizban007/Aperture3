#ifndef _FIELD_SOLVER_LOG_SPH_H_
#define _FIELD_SOLVER_LOG_SPH_H_

#include "data/grid_log_sph_dev.h"
#include "field_solver_dev.h"

namespace Aperture {

class FieldSolver_LogSph : public FieldSolverDev {
 public:
  FieldSolver_LogSph(const Grid_LogSph_dev& g);
  virtual ~FieldSolver_LogSph();
  virtual void update_fields(cu_sim_data& data, double dt,
                             double time = 0.0) override;
  void update_fields(vfield_t& E, vfield_t& B, const vfield_t& J,
                     double dt, double time = 0.0);
  // virtual void compute_flux(const vfield_t& f, sfield_t& flux)
  // override;

  void compute_E_update(vfield_t& E, const vfield_t& B,
                        const vfield_t& J, double dt);
  void compute_B_update(vfield_t& B, const vfield_t& E, double dt);
  virtual void set_background_j(const vfield_t& J) override;
  void boundary_conditions(cu_sim_data& data, double omega);
  void clean_divergence(cu_sim_data& data);
  void init_dev_rho(cu_sim_data& data);

  sfield_t& get_divE() { return m_divE; }
  sfield_t& get_divB() { return m_divB; }

 private:
  const Grid_LogSph_dev& m_grid;
  sfield_t m_divE, m_divB, m_phi_e;
  cudaPitchedPtr* m_dev_rho = nullptr;
  bool m_rho_initialized = false;
};  // ----- end of class FieldSolver_LogSph : public FieldSolver -----

}  // namespace Aperture

#endif  // _FIELD_SOLVER_LOG_SPH_H_
