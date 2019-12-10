#ifndef _GRID_LOG_SPH_PTRS_H_
#define _GRID_LOG_SPH_PTRS_H_

#include "grids/grid_log_sph.h"
#include "cuda/utils/pitchptr.h"

namespace Aperture {

struct mesh_ptrs_log_sph {
  pitchptr<Scalar> l1_e, l2_e, l3_e;
  pitchptr<Scalar> l1_b, l2_b, l3_b;
  pitchptr<Scalar> A1_e, A2_e, A3_e;
  pitchptr<Scalar> A1_b, A2_b, A3_b;
  pitchptr<Scalar> dV;
};

mesh_ptrs_log_sph get_mesh_ptrs(Grid_LogSph& grid);

}  // namespace Aperture

#endif  // _GRID_LOG_SPH_PTRS_H_
