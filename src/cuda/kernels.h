#ifndef _KERNELS_H_
#define _KERNELS_H_
#include <cinttypes>

namespace Aperture {

void compute_tile(uint32_t* tile, const uint32_t* cell, size_t num);

void erase_ptc_in_guard_cells(uint32_t* cell, size_t num);

}  // namespace Aperture

#endif  // _KERNELS_H_
