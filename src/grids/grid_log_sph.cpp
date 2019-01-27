#include "grids/grid_log_sph.h"

namespace Aperture {

template class Grid_LogSph_base<Grid_LogSph>;

Grid_LogSph::Grid_LogSph() {}

Grid_LogSph::~Grid_LogSph() {}

void
Grid_LogSph::init( const SimParams &params ) {
  Grid::init(params);
}

}  // namespace Aperture
