#ifndef _KERNELS_H_
#define _KERNELS_H_
#include "core/enum_types.h"
#include <cinttypes>
#include <curand_kernel.h>

namespace Aperture {

void compute_tile(uint32_t* tile, const uint32_t* cell, size_t num);

void erase_ptc_in_guard_cells(uint32_t* cell, size_t num);

void compute_energy_histogram(uint32_t* hist, const Scalar* E,
                              size_t num, int num_bins, Scalar Emax);

void compute_energy_histogram(uint32_t* hist, const Scalar* E,
                              size_t num, int num_bins, Scalar Emax,
                              const uint32_t* flags, ParticleFlag flag);
void init_rand_states(curandState* states, int seed, int blockPerGrid,
                      int threadPerBlock);
void map_tracked_ptc(uint32_t* flags, uint32_t* cells, size_t num,
                     uint32_t* tracked_map, uint32_t* num_tracked,
                     uint64_t max_tracked);
void adjust_cell_number(uint32_t* cells, size_t num, int shift);

}  // namespace Aperture

#endif  // _KERNELS_H_
