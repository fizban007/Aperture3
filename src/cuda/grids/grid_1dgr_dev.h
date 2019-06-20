#ifndef _GRID_1DGR_DEV_H_
#define _GRID_1DGR_DEV_H_

#include "core/grid.h"
#include "cuda/data/array.h"

namespace Aperture {

class Grid_1dGR_dev : public Grid {
 public:
  Grid_1dGR_dev();
  virtual ~Grid_1dGR_dev();

  virtual void init(const SimParams &params) override;

  struct mesh_ptrs {
    const Scalar *D1, *D2, *D3, *alpha;
    const Scalar *K1, *K1_j, *j0, *agrr, *agrf, *rho0;
    const Scalar *gamma_rr, *gamma_ff, *beta_phi, *B3B1;
  };

  mesh_ptrs get_mesh_ptrs() const;

 private:
  cu_array<Scalar> m_D1, m_D2, m_D3, m_alpha;
  cu_array<Scalar> m_K1, m_K1_j, m_j0, m_agrr, m_agrf, m_rho0;
  cu_array<Scalar> m_gamma_rr, m_gamma_ff, m_beta_phi, m_B3B1;
  // Scalar *m_g1 = nullptr,

};  // ----- end of class Grid_1dGR_dev : public Grid -----

}  // namespace Aperture

#endif  // _GRID_1DGR_DEV_H_
