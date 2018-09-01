#include "data/grid_1dGR.h"

namespace Aperture {

Grid_1dGR::Grid_1dGR() {}

Grid_1dGR::Grid_1dGR(int N) {}

Grid_1dGR::init(const SimParams& params) {
  Grid::init(params);

  // Initialize D1, D2, D3, alpha_grr, and A
  m_D1.resize(m_mesh.dims[0]); m_D1.assign(0.0);
  m_D2.resize(m_mesh.dims[0]); m_D2.assign(1.0);
  m_D3.resize(m_mesh.dims[0]); m_D3.assign(0.0);
  m_alpha_grr.resize(m_mesh.dims[0]); m_alpha_grr.assign(1.0);
  m_A.resize(m_mesh.dims[0]); m_A.assign(1.0);
  m_a2.resize(m_mesh.dims[0]); m_a2.assign(1.0);
  m_angle.resize(m_mesh.dims[0]); m_angle.assign(0.0);

}

Grid_1dGR::const_mesh_ptrs
Grid_1dGR::get_mesh_ptrs() const {
  const_mesh_ptrs result;
  result.D1 = m_D1.data();
  result.D2 = m_D2.data();
  result.D3 = m_D3.data();
  result.alpha_grr = m_alpha_grr.data();
  result.A = m_A.data();
  result.a2 = m_a2.data();
  result.angle = m_angle.data();
  return result;
}


}
