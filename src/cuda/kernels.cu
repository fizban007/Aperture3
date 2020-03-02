#include "cuda/constant_mem.h"
#include "cuda/cudaUtility.h"
#include "cuda/kernels.h"
#include "data/particle_data.h"
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
                uint32_t* tracked_map, uint32_t* num_tracked,
                uint64_t max_tracked) {
  for (size_t n = threadIdx.x + blockIdx.x * blockDim.x; n < num;
       n += blockDim.x * gridDim.x) {
    if (check_bit(flags[n], ParticleFlag::tracked) &&
        cells[n] != MAX_CELL) {
      uint32_t nt = atomicAdd(num_tracked, 1u);
      if (nt < max_tracked) {
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

__global__ void
compute_target_buffers(const uint32_t* cells, size_t num,
                       int* buffer_num, size_t* idx) {
  for (size_t n = threadIdx.x + blockIdx.x * blockDim.x; n < num;
       n += blockDim.x * gridDim.x) {
    if (cells[n] == MAX_CELL) continue;
    size_t zone = dev_mesh.find_zone(cells[n]);

    size_t pos = atomicAdd(&buffer_num[zone], 1);
    // Zone is less than 32, so we can use 5 bits to represent this. The
    // rest of the bits go to encode the index of this particle in that
    // zone.
    idx[n] = ((zone & 0b11111) << (sizeof(size_t) * 8 - 5)) + pos;
    // printf("computed zone is %lu, idx[n] is %lu, num[zone] is %d\n", zone,
           // idx[n], buffer_num[zone]);
  }
}

template <typename DataType>
__global__ void
copy_component_to_buffer(DataType ptc_data, size_t num, size_t* idx,
                         DataType* ptc_buffers) {
  int zone_offset = 0;
  if (dev_mesh.dim() == 2)
    zone_offset = 9;
  else if (dev_mesh.dim() == 1)
    zone_offset = 12;
  int bitshift_width = (sizeof(size_t) * 8 - 5);
  for (size_t n = threadIdx.x + blockIdx.x * blockDim.x; n < num;
       n += blockDim.x * gridDim.x) {
    if (ptc_data.cell[n] == MAX_CELL) continue;
    size_t i = idx[n];
    size_t zone = ((i >> bitshift_width) & 0b11111);
    if (zone == 13) continue;
    size_t pos = i & ((1 << bitshift_width) - 1);
    // printf("zone - offset is %lu, pos is %lu\n", zone - zone_offset, pos);
    // Copy the particle data from ptc_data[n] to ptc_buffers[zone][pos]
    assign_ptc(ptc_buffers[zone - zone_offset], pos, ptc_data, n);
    // printf("ptc_buffers[zone-zone_offset].cell[pos] is %u\n",
    //        ptc_buffers[zone - zone_offset].cell[pos]);
    // Set the particle to empty
    ptc_data.cell[n] = MAX_CELL;
    // Compute particle cell delta
    int dz = (zone / 9) - 1;
    int dy = (zone / 3) % 3 - 1;
    int dx = zone % 3 - 1;
    uint32_t dcell = -dz * dev_mesh.reduced_dim(2) * dev_mesh.dims[0] *
                         dev_mesh.dims[1] -
                     dy * dev_mesh.reduced_dim(1) * dev_mesh.dims[0] -
                     dx * dev_mesh.reduced_dim(0);
    ptc_buffers[zone - zone_offset].cell[pos] += dcell;
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
compute_energy_histogram(uint32_t* hist, const Scalar* E, size_t num,
                         int num_bins, Scalar Emax) {
  Kernels::compute_energy_histogram<<<512, 512>>>(hist, E, num,
                                                  num_bins, Emax);
  // Wait for GPU to finish
  cudaDeviceSynchronize();
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
                uint32_t* tracked_map, uint32_t* num_tracked,
                uint64_t max_tracked) {
  int block_num = std::min(1024ul, (num + 511) / 512);
  Kernels::map_tracked_ptc<<<block_num, 512>>>(
      flags, cells, num, tracked_map, num_tracked, max_tracked);
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

void
compute_target_buffers(const uint32_t* cells, size_t num,
                       int* buffer_num, size_t* idx) {
  Kernels::compute_target_buffers<<<256, 512>>>(cells, num, buffer_num,
                                                idx);
  CudaCheckError();
}

template <typename DataType>
void
copy_ptc_to_buffers(DataType ptc_data, size_t num, size_t* idx,
                    DataType* ptc_buffers) {
  Kernels::copy_component_to_buffer<<<256, 512>>>(ptc_data, num, idx,
                                                  ptc_buffers);
  CudaCheckError();
}

template void copy_ptc_to_buffers<particle_data>(
    particle_data ptc_data, size_t num, size_t* idx,
    particle_data* ptc_buffers);

template void copy_ptc_to_buffers<photon_data>(
    photon_data ptc_data, size_t num, size_t* idx,
    photon_data* ptc_buffers);

}  // namespace Aperture
