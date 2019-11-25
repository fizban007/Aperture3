#ifndef _GRID_1DGAP_H_
#define _GRID_1DGAP_H_

#include "core/grid.h"
#include "core/array.h"

namespace Aperture {

class Grid_1dGap : public Grid {
 public:
  Grid_1dGap();
  Grid_1dGap(int N);
  virtual ~Grid_1dGap();

  virtual void compute_coef(const SimParams& params) override;

  // struct mesh_ptrs {
  //   const Scalar *D1, *D2, *D3, *alpha, *theta;
  //   const Scalar *K1, *K1_j, *j0, *agrr, *agrf, *rho0;
  //   const Scalar *gamma_rr, *gamma_ff, *beta_phi, *B3B1;
  // };

  // mesh_ptrs get_mesh_ptrs() const;
  Scalar j0() const { return m_j0; }
  Scalar rho0() const { return m_rho0; }

 private:
  Scalar m_j0, m_rho0;
  // array<Scalar> m_D1, m_D2, m_D3, m_alpha, m_theta;
  // array<Scalar> m_K1, m_K1_j, m_j0, m_agrr, m_agrf, m_rho0;
  // array<Scalar> m_gamma_rr, m_gamma_ff, m_beta_phi, m_B3B1;
};

}

#endif // _GRID_1DGAP_H_
