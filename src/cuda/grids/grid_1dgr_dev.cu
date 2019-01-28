#include "cuda/cudaUtility.h"
#include "cuda/grids/grid_1dgr_dev.h"

namespace Aperture {

Grid_1dGR_dev::Grid_1dGR_dev() {}

Grid_1dGR_dev::~Grid_1dGR_dev() {}

void
Grid_1dGR_dev::init(const SimParams& params) {
  Grid::init(params);

  m_D1.resize(params.N[0] + 2 * params.guard[0]);
  m_D2.resize(params.N[0] + 2 * params.guard[0]);
  m_D3.resize(params.N[0] + 2 * params.guard[0]);
  m_alpha2.resize(params.N[0] + 2 * params.guard[0]);
  m_dPdt.resize(params.N[0] + 2 * params.guard[0]);
  m_Btp.resize(params.N[0] + 2 * params.guard[0]);
  m_agrr.resize(params.N[0] + 2 * params.guard[0]);
  m_agrf.resize(params.N[0] + 2 * params.guard[0]);
  m_g5.resize(params.N[0] + 2 * params.guard[0]);

  // TODO: initialize all the array content
}

Grid_1dGR_dev::mesh_ptrs
Grid_1dGR_dev::get_mesh_ptrs() const {
  mesh_ptrs ptrs;

  ptrs.D1 = m_D1.data_d();
  ptrs.D2 = m_D2.data_d();
  ptrs.D3 = m_D3.data_d();
  ptrs.alpha2 = m_alpha2.data_d();
  ptrs.dPdt = m_dPdt.data_d();
  ptrs.Btp = m_Btp.data_d();
  ptrs.agrr = m_agrr.data_d();
  ptrs.agrf = m_agrf.data_d();

  return ptrs;
}

}  // namespace Aperture