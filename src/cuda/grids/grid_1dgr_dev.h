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
    const Scalar *D1, *D2, *D3, *alpha2;
    const Scalar *dPdt, *Btp, *agrr, *agrf, *rho0;
  };

  mesh_ptrs get_mesh_ptrs() const;

 private:
  Array<Scalar> m_D1, m_D2, m_D3, m_alpha2;
  Array<Scalar> m_dPdt, m_Btp, m_agrr, m_agrf, m_rho0;
  // Scalar *m_g1 = nullptr,

};  // ----- end of class Grid_1dGR_dev : public Grid -----

}  // namespace Aperture

#endif  // _GRID_1DGR_DEV_H_
