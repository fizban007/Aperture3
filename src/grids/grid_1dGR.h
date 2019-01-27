#ifndef _GRID_1DGR_H_
#define _GRID_1DGR_H_

#include "core/grid.h"

namespace Aperture {

class Grid_1dGR : public Grid {
 public:
  Grid_1dGR();
  Grid_1dGR(int N);
  virtual ~Grid_1dGR();

  void init(const SimParams& params);

  Scalar* D1() { return m_D1.data(); }
  Scalar* D2() { return m_D2.data(); }
  Scalar* D3() { return m_D3.data(); }
  Scalar* alpha_grr() { return m_alpha_grr.data(); }
  Scalar* A() { return m_A.data(); }
  Scalar* a2() { return m_a2.data(); }
  Scalar* angle() { return m_angle.data(); }

  struct mesh_ptrs {
    Scalar *D1, *D2, *D3;
    Scalar *A, *alpha_grr, *angle;
    Scalar* a2;
  };
  struct const_mesh_ptrs {
    const Scalar *D1, *D2, *D3;
    const Scalar *A, *alpha_grr, *angle;
    const Scalar* a2;
  };
  const_mesh_ptrs get_mesh_ptrs() const;

 private:
  cu_multi_array<Scalar> m_D1, m_D2, m_D3;
  cu_multi_array<Scalar> m_A, m_alpha_grr, m_angle;
  cu_multi_array<Scalar> m_a2;
};  // ----- end of class Grid_1dGR : public Grid -----

}  // namespace Aperture

#endif  // _GRID_1DGR_H_
