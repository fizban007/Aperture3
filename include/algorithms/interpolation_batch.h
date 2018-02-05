#ifndef _INTERPOLATION_BATCH_H_
#define _INTERPOLATION_BATCH_H_

#include <cmath>
#include <vector>
#include <immintrin.h>
#include "data/grid.h"
#include "cuda/cuda_control.h"

namespace Aperture {

class interpolator_batch
{
 public:
  interpolator_batch(Grid& grid, int order = 1) : m_grid(grid), m_order(order) {}
  virtual ~interpolator_batch() {}

 private:

  int m_order;
  Grid& m_grid;
}; // ----- end of class interpolator -----

}

#endif  // _INTERPOLATION_BATCH_H_
