#ifndef _GRID_LOG_SPH_DEV_H_
#define _GRID_LOG_SPH_DEV_H_

#include "cuda/data/fields_dev.h"
#include "cuda/utils/pitchptr.cuh"
#include "grids/grid_log_sph_base.h"

namespace Aperture {

class Grid_LogSph_dev : public Grid_LogSph_base<Grid_LogSph_dev> {
public:
  typedef Grid_LogSph_base<Grid_LogSph_dev> base_class;

  Grid_LogSph_dev();
  virtual ~Grid_LogSph_dev();

  virtual void init(const SimParams &params) override;

  struct mesh_ptrs {
    pitchptr<Scalar> l1_e, l2_e, l3_e;
    pitchptr<Scalar> l1_b, l2_b, l3_b;
    pitchptr<Scalar> A1_e, A2_e, A3_e;
    pitchptr<Scalar> A1_b, A2_b, A3_b;
    pitchptr<Scalar> dV;
  };

  mesh_ptrs get_mesh_ptrs() const;

  void compute_flux(cu_scalar_field<Scalar> &flux, cu_vector_field<Scalar> &B,
                    cu_vector_field<Scalar> &B_bg) const;

  Scalar alpha(Scalar r, Scalar rs) const { return std::sqrt(1.0 - rs / r); }
  Scalar l1(Scalar r, Scalar rs) const {
    Scalar a = alpha(r, rs);
    return r * a + 0.5 * rs * std::log(2.0 * r * (1.0 + a) - rs);
  }
  Scalar A2(Scalar r, Scalar rs) const {
    Scalar a = alpha(r, rs);
    return 0.25 * r * a * (2.0 * r + 3.0 * rs) +
           0.375 * rs * rs * std::log(2.0 * r * (1.0 + a) - rs);
  }
  Scalar V3(Scalar r, Scalar rs) const {
    Scalar a = alpha(r, rs);
    return r * a * (8.0 * r * r + 10.0 * r * rs + 15.0 * rs * rs) / 24.0 +
        0.3125 * rs * rs * rs * std::log(2.0 * r * (1.0 + a) - rs);
  }

private:
  cu_multi_array<Scalar> m_l1_e, m_l2_e, m_l3_e;
  cu_multi_array<Scalar> m_l1_b, m_l2_b, m_l3_b;
  cu_multi_array<Scalar> m_A1_e, m_A2_e, m_A3_e;
  cu_multi_array<Scalar> m_A1_b, m_A2_b, m_A3_b;
  cu_multi_array<Scalar> m_dV;
};

} // namespace Aperture

#endif // _GRID_LOG_SPH_DEV_H_
