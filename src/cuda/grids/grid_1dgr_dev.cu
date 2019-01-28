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
  m_dPdt.resize(params.N[0] + 2 * params.guard[0]);
  m_Btp.resize(params.N[0] + 2 * params.guard[0]);
  m_agrr.resize(params.N[0] + 2 * params.guard[0]);
  m_agrf.resize(params.N[0] + 2 * params.guard[0]);
  m_g5.resize(params.N[0] + 2 * params.guard[0]);

  // TODO: initialize all the array content
}

}  // namespace Aperture