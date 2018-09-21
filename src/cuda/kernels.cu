#include "cuda/constant_mem.h"
#include "cuda/cudaUtility.h"
#include "cuda/kernels.h"

namespace Aperture {

namespace Kernels {

__global__ void
compute_tile(uint32_t* tile, const uint32_t* cell, size_t num) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num;
       i += blockDim.x * gridDim.x) {
    // tile[i] = cell[i] / dev_mesh.tileSize[0];
    if (i < num) tile[i] = dev_mesh.tile_id(cell[i]);
  }
}

__global__ void
erase_ptc_in_guard_cells(uint32_t* cell, size_t num) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num;
       i += blockDim.x * gridDim.x) {
    // if (i == 0) printf("num is %d\n", num);
    if (i < num) {
      auto c = cell[i];
      if (!dev_mesh.is_in_bulk(c))
        // if (c < dev_mesh.guard[0] || c >= dev_mesh.dims[0] -
        // dev_mesh.guard[0])
        cell[i] = MAX_CELL;
    }
  }
}

__global__ void
compute_energy_histogram(uint32_t* hist, const Scalar* E, size_t num, int num_bins, Scalar E_max) {
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

}  // namespace Kernels

void
compute_tile(uint32_t* tile, const uint32_t* cell, size_t num) {
  Kernels::compute_tile<<<256, 256>>>(tile, cell, num);
  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  CudaCheckError();
}

void
erase_ptc_in_guard_cells(uint32_t* cell, size_t num) {
  Kernels::erase_ptc_in_guard_cells<<<512, 512>>>(cell, num);
  // Wait for GPU to finish
  cudaDeviceSynchronize();
  CudaCheckError();
}

void
compute_energy_histogram(uint32_t* hist, const Scalar* E, size_t num, int num_bins, Scalar Emax) {
  Kernels::compute_energy_histogram<<<512, 512>>>(hist, E, num, num_bins, Emax);
  // Wait for GPU to finish
  cudaDeviceSynchronize();
  CudaCheckError();
}

}  // namespace Aperture