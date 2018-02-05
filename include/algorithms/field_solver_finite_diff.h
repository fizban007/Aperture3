#ifndef _FIELD_SOLVER_FINITE_DIFF_H_
#define _FIELD_SOLVER_FINITE_DIFF_H_

#include "field_solver.h"

namespace Aperture {

class FieldSolver_FiniteDiff : public FieldSolver {
 public:
  FieldSolver_FiniteDiff(const Grid& g, double alpha = 0.0);
  virtual ~FieldSolver_FiniteDiff();

  virtual void update_fields(vfield_t& E, vfield_t& B, const vfield_t& J, double dt,
                             double time = 0.0);
  virtual void update_fields(SimData& data, double dt, double time = 0.0);

  virtual void compute_flux(const vfield_t& f, sfield_t& flux);

  void compute_E_update(vfield_t& E, const vfield_t& B, const vfield_t& J, double dt);
  void compute_B_update(vfield_t& B, const vfield_t& E, const vfield_t& J, double dt);
  void compute_laplacian(const vfield_t& input, vfield_t& output,
                         const Index& start, const Extent& ext,
                         const bool is_boundary[], int order = 2);
  void compute_laplacian(const sfield_t& input, sfield_t& output,
                         const Index& start, const Extent& ext,
                         const bool is_boundary[], int order = 2);

 private:
  // FiniteDiff m_fd;

  vfield_t m_dE, m_dB;
  vfield_t m_E_tmp, m_B_old;
  // These are two tmp fields for Laplacian
  vfield_t m_vfield_tmp;
  sfield_t m_sfield_tmp;

  double m_alpha, m_beta;
};  // ----- end of class field_updater_finite_diff -----
}

#endif  // _FIELD_SOLVER_FINITE_DIFF_H_
