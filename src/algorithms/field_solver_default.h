#ifndef _FIELD_SOLVER_DEFAULT_H_
#define _FIELD_SOLVER_DEFAULT_H_

#include "field_solver.h"
#include "data/grid.h"

namespace Aperture {

class field_solver_default : public field_solver {
 public:
  field_solver_default(const Grid& g);
  virtual ~field_solver_default();
  virtual void update_fields(sim_data& data, double dt,
                             double time = 0.0) override;
  void update_fields(vfield_t& E, vfield_t& B, const vfield_t& J,
                     double dt, double time = 0.0);
  // virtual void compute_flux(const vfield_t& f, sfield_t& flux)
  // override;

  void compute_E_update(vfield_t& E, const vfield_t& B,
                        const vfield_t& J, double dt);
  void compute_B_update(vfield_t& B, const vfield_t& E, double dt);

 private:
  vfield_t m_dE, m_dB;
};  // ----- end of class field_solver_default : public field_solver -----

}  // namespace Aperture

#endif  // _FIELD_SOLVER_DEFAULT_H_
