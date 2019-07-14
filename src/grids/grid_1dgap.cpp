#include "grid_1dgap.h"
#include "utils/logger.h"
#include "utils/util_functions.h"

namespace Aperture {


Grid_1dGap::Grid_1dGap() : Grid() {}

Grid_1dGap::Grid_1dGap(int N) : Grid(N, 1, 1) {}

Grid_1dGap::~Grid_1dGap() {}

void
Grid_1dGap::compute_coef(const SimParams &params) {
  m_j0 = params.B0;
  // TODO: introduce parameter for this k_rho
  m_rho0 = 0.5 * m_j0;
}


}
