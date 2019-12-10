#include "cuda/constant_mem.h"
#include "cuda/cudaUtility.h"
#include "cuda/kernels.h"
#include "utils/util_functions.h"

namespace Aperture {

namespace Kernels {

__global__ void
init_rand_states(curandState* states, int seed) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  curand_init(seed, id, 0, &states[id]);
}

__global__ void
compute_tile(uint32_t* tile, const uint32_t* cell, size_t num) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num;
       i += blockDim.x * gridDim.x) {
    if (i < num) tile[i] = dev_mesh.tile_id(cell[i]);
  }
}

__global__ void
erase_ptc_in_guard_cells(uint32_t* cell, size_t num) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num;
       i += blockDim.x * gridDim.x) {
    if (i < num) {
      auto c = cell[i];
      if (!dev_mesh.is_in_bulk(c)) cell[i] = MAX_CELL;
    }
  }
}

__global__ void
compute_energy_histogram(uint32_t* hist, const Scalar* E, size_t num,
                         int num_bins, Scalar E_max) {
  Scalar dlogE = std::log(E_max) / (Scalar)num_bins;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num;
       i += blockDim.x * gridDim.x) {
    if (i < num) {
      Scalar logE = std::log(E[i]);
      int idx = (int)floorf(logE / dlogE);
      if (idx < 0) idx = 0;
      if (idx >= num_bins) idx = num_bins - 1;

      atomicAdd(&hist[idx], 1);
    }
  }
}
__global__ void
compute_energy_histogram(uint32_t* hist, const Scalar* E, size_t num,
                         int num_bins, Scalar E_max,
                         const uint32_t* flags, ParticleFlag flag) {
  Scalar dlogE = std::log(E_max) / (Scalar)num_bins;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num;
       i += blockDim.x * gridDim.x) {
    if (i < num) {
      if (!check_bit(flags[i], flag)) continue;
      Scalar logE = std::log(E[i]);
      int idx = (int)floorf(logE / dlogE);
      if (idx < 0) idx = 0;
      if (idx >= num_bins) idx = num_bins - 1;

      atomicAdd(&hist[idx], 1);
    }
  }
}

__global__ void
map_tracked_ptc(uint32_t* flags, uint32_t* cells, size_t num,
                uint32_t* tracked_map, uint32_t* num_tracked) {
  for (size_t n = threadIdx.x + blockIdx.x * blockDim.x; n < num;
       n += blockDim.x * gridDim.x) {
    if (check_bit(flags[n], ParticleFlag::tracked) &&
        cells[n] != MAX_CELL) {
      uint32_t nt = atomicAdd(num_tracked, 1u);
      if (nt < MAX_TRACKED) {
        tracked_map[nt] = n;
      }
    }
  }
}

__global__ void
adjust_cell_number(uint32_t* cells, size_t num, int shift) {
  for (size_t n = threadIdx.x + blockIdx.x * blockDim.x; n < num;
       n += blockDim.x * gridDim.x) {
    cells[n] += shift;
  }
}

}  // namespace Kernels

void
compute_tile(uint32_t* tile, const uint32_t* cell, size_t num) {
  Kernels::compute_tile<<<256, 256>>>(tile, cell, num);
  // Wait for GPU to finish before accessing on host
  // cudaDeviceSynchronize();
  CudaCheckError();
}

void
erase_ptc_in_guard_cells(uint32_t* cell, size_t num) {
  Kernels::erase_ptc_in_guard_cells<<<512, 512>>>(cell, num);
  // Wait for GPU to finish
  // cudaDeviceSynchronize();
  CudaCheckError();
}

void
compute_energy_histogram(uint32_t* hist, const Scalar* E, size_t num,
                         int num_bins, Scalar Emax) {
  Kernels::compute_energy_histogram<<<512, 512>>>(hist, E, num,
                                                  num_bins, Emax);
  // Wait for GPU to finish
  // cudaDeviceSynchronize();
  CudaCheckError();
}

void
compute_energy_histogram(uint32_t* hist, const Scalar* E, size_t num,
                         int num_bins, Scalar Emax,
                         const uint32_t* flags, ParticleFlag flag) {
  Kernels::compute_energy_histogram<<<512, 512>>>(
      hist, E, num, num_bins, Emax, flags, flag);
  // Wait for GPU to finish
  // cudaDeviceSynchronize();
  CudaCheckError();
}

void
init_rand_states(curandState* states, int seed, int blockPerGrid,
                 int threadPerBlock) {
  Kernels::init_rand_states<<<blockPerGrid, threadPerBlock>>>(states,
                                                              seed);
  CudaCheckError();
}

void
map_tracked_ptc(uint32_t* flags, uint32_t* cells, size_t num,
                uint32_t* tracked_map, uint32_t* num_tracked) {
  int block_num = std::min(1024ul, (num + 511) / 512);
  Kernels::map_tracked_ptc<<<block_num, 512>>>(flags, cells, num, tracked_map,
                                               num_tracked);
  CudaCheckError();
  CudaSafeCall(cudaDeviceSynchronize());
}

void
adjust_cell_number(uint32_t* cells, size_t num, int shift) {
  int block_num = std::min(1024ul, (num + 511) / 512);
  Kernels::adjust_cell_number<<<block_num, 512>>>(cells, num, shift);
  CudaCheckError();
  CudaSafeCall(cudaDeviceSynchronize());
}

}  // namespace Aperture
