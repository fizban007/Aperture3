#ifndef _GRID_LOG_SPH_DEV_H_
#define _GRID_LOG_SPH_DEV_H_

#include "data/grid_log_sph_base.h"
#include "data/fields_dev.h"

namespace Aperture {

class Grid_LogSph_dev : public Grid_LogSph_base<Grid_LogSph_dev> {
 public:
  typedef Grid_LogSph_base<Grid_LogSph_dev> base_class;

  Grid_LogSph_dev();
  virtual ~Grid_LogSph_dev();

  void init(const SimParams& params) override;

  struct mesh_ptrs {
    cudaPitchedPtr l1_e, l2_e, l3_e;
    cudaPitchedPtr l1_b, l2_b, l3_b;
    cudaPitchedPtr A1_e, A2_e, A3_e;
    cudaPitchedPtr A1_b, A2_b, A3_b;
    cudaPitchedPtr dV;
  };

  mesh_ptrs get_mesh_ptrs() const;

  void compute_flux(cu_scalar_field<Scalar>& flux,
                    cu_vector_field<Scalar>& B,
                    cu_vector_field<Scalar>& B_bg) const;

 private:
  cu_multi_array<Scalar> m_l1_e, m_l2_e, m_l3_e;
  cu_multi_array<Scalar> m_l1_b, m_l2_b, m_l3_b;
  cu_multi_array<Scalar> m_A1_e, m_A2_e, m_A3_e;
  cu_multi_array<Scalar> m_A1_b, m_A2_b, m_A3_b;
  cu_multi_array<Scalar> m_dV;
};

}  // namespace Aperture

#endif  // _GRID_LOG_SPH_DEV_H_
