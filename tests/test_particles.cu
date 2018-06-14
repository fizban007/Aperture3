#include "data/particles.h"
#include "utils/timer.h"
#include "utils/logger.h"
#include <cuda.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <random>
#include "catch.hpp"
#include "cub/cub.cuh"

using namespace Aperture;

__global__
void set_cells(uint32_t* cells) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  cells[idx] = idx;
}

TEST_CASE("Initializing and Adding particles", "[Particles]") {
  size_t N = 100000;
  Particles ptc(N);

  CHECK(ptc.number() == 0);
  CHECK(ptc.numMax() == N);

  ptc.append(0.5, 1.0, 100, ParticleType::electron);
  ptc.append(0.2, 1.0, 200, ParticleType::electron);
  ptc.append(0.2, 0.0, 300, ParticleType::positron);
  ptc.append(0.2, -1.0, 400, ParticleType::positron);

  CHECK(ptc.number() == 4);
  CHECK(ptc.data().x1[0] == Approx(0.5));
  CHECK(ptc.data().p1[3] == -1.0);
  CHECK(ptc.data().cell[2] == 300);
  CHECK(ptc.check_type(2) == ParticleType::positron);
  CHECK(ptc.check_type(0) == ParticleType::electron);

  set_cells<<<256, 256>>>(ptc.data().cell);
  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  for (size_t i = 0; i < 256*256; i++) {
    CHECK(ptc.data().cell[i] == i);
  }
}

TEST_CASE("Testing memory allocation", "[Particles]") {
  size_t N = 10000000;
  size_t free_mem, total_mem;
  cudaMemGetInfo(&free_mem, &total_mem);
  Logger::print_info("Current free memory is {}MB", (double)free_mem / (1024*1024));
  Particles ptc(N);
  cudaMemPrefetchAsync(ptc.tmp_data(), N*sizeof(double), 0);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  cudaMemGetInfo(&free_mem, &total_mem);
  Logger::print_info("Current free memory is {}MB", (double)free_mem / (1024*1024));
}

TEST_CASE("Sorting particles, trivial", "[Particles]") {
  size_t N = 10000000;
  Particles ptc(N);

  std::default_random_engine gen;
  std::uniform_int_distribution<int> dist(-1, 1);

  for (size_t i = 0; i < N; i++) {
    ptc.data().x1[i] = (float)i;
    ptc.data().p1[i] = (float)(N - i);
    ptc.data().cell[i] = i / 64 + dist(gen);
  }
  ptc.set_num(N);
  ptc.sync_to_device(0);

  timer::stamp();
  auto ptr_cell = thrust::device_pointer_cast(ptc.data().cell);
  auto ptr_x1 = thrust::device_pointer_cast(ptc.data().x1);

  thrust::sort_by_key(ptr_cell, ptr_cell + N, ptr_x1);
  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  timer::show_duration_since_stamp("sorting by tile on gpu", "ms");
}

TEST_CASE("Sorting particles, using cub", "[Particles]") {
  size_t N = 10000000;
  Particles ptc(N);
  std::default_random_engine gen;
  std::uniform_int_distribution<int> dist(-1, 1);

  for (size_t i = 0; i < N; i++) {
    ptc.data().x1[i] = (float)i;
    ptc.data().p1[i] = (float)(N - i);
    ptc.data().cell[i] = i / 64 + dist(gen);
  }
  ptc.set_num(N);
  cudaDeviceProp p;
  int deviceId;
  cudaGetDevice(&deviceId);
  cudaGetDeviceProperties(&p, deviceId);
  ptc.sync_to_device(deviceId);

  uint32_t* cell_alt;
  float* x1_alt;
  cudaMalloc(&cell_alt, N * sizeof(uint32_t));
  cudaMalloc(&x1_alt, N * sizeof(float));

  cub::DoubleBuffer<uint32_t> d_keys(ptc.data().cell, cell_alt);
  cub::DoubleBuffer<float> d_values(ptc.data().x1, x1_alt);

  // Determine temporary device storage requirements
  void     *d_temp_storage = NULL;
  size_t   temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, N);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);

  timer::stamp();
  // Run sorting operation
  cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, N);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  timer::show_duration_since_stamp("sort using cub radix sort", "ms");

  cudaFree(d_temp_storage);
  cudaFree(cell_alt);
  cudaFree(x1_alt);
}

TEST_CASE("Sorting Particles by tile", "[Particles]") {
  size_t N = 10000000;
  Particles ptc(N);

  for (size_t i = 0; i < N; i++) {
    ptc.append(0.1, 0.0, i, ParticleType::electron);
  }

  int tile_size = 64;
  ptc.compute_tile_num(tile_size);

  for (size_t i = 0; i < N; i++) {
    // CHECK(ptc.data().tile[i] == i / tile_size);
  }
  ptc.sync_to_device(0);

  timer::stamp();
  // ptc.sort_by_cell();
  ptc.sort_by_tile(tile_size);
  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  timer::show_duration_since_stamp("sorting by cell on gpu, in class", "ms");
}