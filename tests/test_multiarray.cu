#include "catch.hpp"
#include "core/detail/multi_array_utils.hpp"
#include "cuda/cudaUtility.h"
#include "cuda/data/cu_multi_array.h"
#include <cuda_runtime.h>
#include <iostream>

using namespace Aperture;

// __global__
// void add(const Scalar* a, const Scalar* b, Scalar* c);

// __global__
// void add2D(const Extent ext, const Scalar* a, const Scalar* b,
// Scalar* c) {

//   for (int j = blockIdx.y * blockDim.y + threadIdx.y;
//        j < ext.y;
//        j += blockDim.y * gridDim.y) {
//     for (int i = blockIdx.x * blockDim.x + threadIdx.x;
//         i < ext.x;
//         i += blockDim.x * gridDim.x) {
//       size_t idx = i + j * ext.x;
//       c[idx] = a[idx] + b[idx];
//     }
//   }

// }

struct Data {
  cu_multi_array<Scalar> a, b, c;
  size_t size, memSize;

  Data(int x, int y = 1, int z = 1)
      : a(x, y, z), b(x, y, z), c(x, y, z) {
    size = x * y * z;
    memSize = size * sizeof(Scalar);
    a.assign_dev(0.0);
    b.assign_dev(0.0);
    c.assign_dev(0.0);
    cudaDeviceSynchronize();
    a.assign(0.0);
    b.assign(0.0);
    c.assign(0.0);
    // b.sync_to_host();
    // c.sync_to_host();
  }

  void prefetch(int deviceId) {
    // cudaMemPrefetchAsync(a.data(), memSize, deviceId);
    // cudaMemPrefetchAsync(b.data(), memSize, deviceId);
    // cudaMemPrefetchAsync(c.data(), memSize, deviceId);
    a.sync_to_device(deviceId);
    b.sync_to_device(deviceId);
    c.sync_to_device(deviceId);
  }

  void prefetch() {
    // cudaMemPrefetchAsync(c.data(), memSize, cudaCpuDeviceId);
    c.sync_to_host();
  }
};

TEST_CASE("Initialize multi_array", "[MultiArray]") {
  Data data(256, 256);

  data.a.assign_dev(1.0);
  data.b.assign_dev(2.0);

  // // add<<<256, 256>>>(data.a.data(), data.b.data(), data.c.data());
  dim3 blockSize(8, 8, 8);
  dim3 gridSize(8, 8, 8);
  Kernels::map_array_binary_op<Scalar><<<gridSize, blockSize>>>(
      data.a.data_d(), data.b.data_d(), data.c.data_d(),
      data.a.extent(), detail::Op_Plus<Scalar>());
  CudaCheckError();

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  data.c.sync_to_host();

  size_t N = data.a.size();
  for (size_t i = 0; i < N; i++) {
    CHECK(data.c[i] == 3.0f);
  }
}

// TEST_CASE("Add 2D multi_array", "[MultiArray]") {
//   Data data(1500, 1500);

//   data.a.assign(1.0);
//   data.b.assign(2.0);

//   cudaDeviceProp p;
//   int deviceId;
//   cudaGetDevice(&deviceId);
//   cudaGetDeviceProperties(&p, deviceId);
//   data.prefetch(deviceId);

//   dim3 blockSize(32, 32);
//   dim3 gridSize(32, 32);
//   add2D<<<gridSize, blockSize>>>(data.a.extent(), data.a.data(),
//   data.b.data(), data.c.data());

//   data.prefetch();
//   // Wait for GPU to finish before accessing on host
//   cudaDeviceSynchronize();

//   size_t N = data.a.size();
//   for (size_t i = 0; i < N; i++) {
//     CHECK(data.c[i] == 3.0f);
//   }
// }

TEST_CASE("Map Array Multiply", "[MultiArray]") {
  int deviceId;
  cudaGetDevice(&deviceId);
  std::cout << "device is " << deviceId << std::endl;
  using namespace Aperture::detail;
  Data data(150, 150, 100);

  data.a.assign(2.0);
  data.b.assign(1.5);
  std::cout << data.a.extent() << std::endl;

  data.prefetch(deviceId);

  // dim3 blockSize(32, 32);
  // dim3 gridSize(32, 32);
  dim3 blockSize(8, 8, 8);
  dim3 gridSize(16, 16, 8);
  Kernels::map_array_binary_op<Scalar><<<gridSize, blockSize>>>(
      data.a.data_d(), data.b.data_d(), data.c.data_d(),
      data.a.extent(), Op_Multiply<Scalar>());
  CudaCheckError();

  data.prefetch();
  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  size_t N = data.c.size();
  for (size_t i = 0; i < N; i++) {
    INFO("i, j, k are " << i % 150 << ", " << (i / 150) % 150 << ", "
                        << i / (150 * 150));
    REQUIRE(data.c[i] == 3.0f);
  }
}
