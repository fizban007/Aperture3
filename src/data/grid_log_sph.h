#ifndef _GRID_LOG_SPH_H_
#define _GRID_LOG_SPH_H_

#include "data/grid.h"
#include "data/fields_dev.h"

namespace Aperture {

class Grid_LogSph : public Grid
{
 public:
  Grid_LogSph();
  virtual ~Grid_LogSph();

  void init(const SimParams& params);

  struct mesh_ptrs {
    cudaPitchedPtr l1_e, l2_e, l3_e;
    cudaPitchedPtr l1_b, l2_b, l3_b;
    cudaPitchedPtr A1_e, A2_e, A3_e;
    cudaPitchedPtr A1_b, A2_b, A3_b;
    cudaPitchedPtr dV;
  };

  mesh_ptrs get_mesh_ptrs() const;

  void compute_flux(cu_scalar_field<Scalar>& flux, cu_vector_field<Scalar>& B,
                    cu_vector_field<Scalar>& B_bg) const;

 private:
  cu_multi_array<Scalar> m_l1_e, m_l2_e, m_l3_e;
  cu_multi_array<Scalar> m_l1_b, m_l2_b, m_l3_b;
  cu_multi_array<Scalar> m_A1_e, m_A2_e, m_A3_e;
  cu_multi_array<Scalar> m_A1_b, m_A2_b, m_A3_b;
  cu_multi_array<Scalar> m_dV;
}; // ----- end of class Grid_LogSph : public Grid -----


}

#endif  // _GRID_LOG_SPH_H_
