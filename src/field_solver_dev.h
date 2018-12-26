#ifndef _FIELD_SOLVER_DEV_H_
#define _FIELD_SOLVER_DEV_H_

#include <memory>
// #include "algorithms/finite_diff.h"
// #include "boundary_conditions.h"
#include "data/callbacks.h"
#include "data/fields_dev.h"
#include "sim_data_dev.h"

namespace Aperture {

class FieldSolverDev {
 public:
  typedef VectorField<Scalar> vfield_t;
  typedef ScalarField<Scalar> sfield_t;

  FieldSolverDev() {}
  virtual ~FieldSolverDev() {}

  virtual void update_fields(SimData& data, double dt,
                             double time = 0.0) = 0;

  virtual void set_background_j(const vfield_t& j) = 0;

  // virtual void compute_E_update(vfield_t& E, const vfield_t& B, const
  // vfield_t& J, double dt) = 0; virtual void
  // compute_B_update(vfield_t& B, const vfield_t& E, double dt) = 0;

  // virtual void compute_flux(const vfield_t& f, sfield_t& flux) = 0;

  // void set_boundary_condition(const BoundaryConditions& bc) { m_bc =
  // &bc; }

  void register_comm_callback(const vfield_comm_callback& callback) {
    m_comm_callback_vfield = callback;
  }
  void register_comm_callback(const sfield_comm_callback& callback) {
    m_comm_callback_sfield = callback;
  }
  // void register_bc_callback(const fieldBC_callback& callback) {
  //   m_bc_callback = callback;
  // }

 protected:
  vfield_comm_callback m_comm_callback_vfield;
  sfield_comm_callback m_comm_callback_sfield;

  // External boundary condition, memory managed elsewhere
  // const BoundaryConditions* m_bc = nullptr;
};  // ----- end of class field_solver -----
}  // namespace Aperture

#endif  // _FIELD_SOLVER_H_
