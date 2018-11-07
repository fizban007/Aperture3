#ifndef _KERNELS_H_
#define _KERNELS_H_
#include "data/enum_types.h"
#include <curand_kernel.h>
#include <cinttypes>

namespace Aperture {

void compute_tile(uint32_t* tile, const uint32_t* cell, size_t num);

void erase_ptc_in_guard_cells(uint32_t* cell, size_t num);

void compute_energy_histogram(uint32_t* hist, const Scalar* E,
                              size_t num, int num_bins, Scalar Emax);

void compute_energy_histogram(uint32_t* hist, const Scalar* E,
                              size_t num, int num_bins, Scalar Emax,
                              const uint32_t* flags, ParticleFlag flag);
void init_rand_states(curandState* states, int seed, int threadPerBlock,
                      int blockPerGrid);

}  // namespace Aperture

#endif  // _KERNELS_H_
