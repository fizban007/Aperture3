#include "data/particles.h"
#include "utils/timer.h"
#include "utils/logger.h"
#include <cuda.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <random>
#include "catch.hpp"

using namespace Aperture;

__global__
void set_cells(uint32_t* cells) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  cells[idx] = idx;
}

TEST_CASE("Initializing and Adding particles", "[Particles]") {
  size_t N = 100000;
  Particles ptc(N);

  CHECK(ptc.type() == ParticleType::electron);
  CHECK(ptc.charge() == -1.0);
  CHECK(ptc.mass() == 1.0);
  CHECK(ptc.number() == 0);
  CHECK(ptc.numMax() == N);

  ptc.append(0.5, 1.0, 100);
  ptc.append(0.2, 1.0, 200);
  ptc.append(0.2, 0.0, 300);
  ptc.append(0.2, -1.0, 400);

  CHECK(ptc.number() == 4);
  CHECK(ptc.data().x1[0] == Approx(0.5));
  CHECK(ptc.data().p1[3] == -1.0);
  CHECK(ptc.data().cell[2] == 300);

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

  timer::stamp();
  auto ptr_cell = thrust::device_pointer_cast(ptc.data().cell);
  auto ptr_x1 = thrust::device_pointer_cast(ptc.data().x1);

  thrust::sort_by_key(ptr_cell, ptr_cell + N, ptr_x1);
  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  timer::show_duration_since_stamp("sorting by tile on gpu", "ms");
}

TEST_CASE("Sorting Particles by tile", "[Particles]") {
  size_t N = 10000000;
  Particles ptc(N);

  for (size_t i = 0; i < N; i++) {
    ptc.append(0.1, 0.0, i);
  }

  int tile_size = 64;
  ptc.compute_tile_num(tile_size);

  for (size_t i = 0; i < N; i++) {
    CHECK(ptc.data().tile[i] == i / tile_size);
  }
  ptc.sync_to_device(0);

  timer::stamp();
  // ptc.sort_by_cell();
  ptc.sort_by_tile(tile_size);
  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  timer::show_duration_since_stamp("sorting by cell on gpu, in class", "ms");
}