#include "cuda/kernels.h"
#include "cuda/constant_mem.h"
#include "cuda/cudaUtility.h"

namespace Aperture {

namespace Kernels {

__global__
void compute_tile(uint32_t* tile, const uint32_t* cell, size_t num) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < num;
       i += blockDim.x * gridDim.x) {
    // tile[i] = cell[i] / dev_mesh.tileSize[0];
    if (i < num)
      tile[i] = dev_mesh.tile_id(cell[i]);
  }
}

__global__
void erase_ptc_in_guard_cells(uint32_t* cell, size_t num) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < num;
       i += blockDim.x * gridDim.x) {
    // if (i == 0) printf("num is %d\n", num);
    if (i < num) {
      auto c = cell[i];
      if (!dev_mesh.is_in_bulk(c))
      // if (c < dev_mesh.guard[0] || c >= dev_mesh.dims[0] - dev_mesh.guard[0])
        cell[i] = MAX_CELL;
    }
  }
}

}

void compute_tile(uint32_t* tile, const uint32_t* cell, size_t num) {
  Kernels::compute_tile<<<256, 256>>>(tile, cell, num);
  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  CudaCheckError();
}

void erase_ptc_in_guard_cells(uint32_t* cell, size_t num) {
  Kernels::erase_ptc_in_guard_cells<<<512,512>>>(cell, num);
  // Wait for GPU to finish
  cudaDeviceSynchronize();
  CudaCheckError();
}

}