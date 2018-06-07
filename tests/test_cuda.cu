#include "data/particle_data.h"
#include <cuda.h>
#include "catch.hpp"

using namespace Aperture;

__global__
void add(const float* a, const float* b, float* c) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  c[i] = a[i] + b[i];
}

TEST_CASE("Launching cuda kernel", "[Cuda]") {
  size_t N = 256*256;
  float *a, *b, *c;
  cudaMallocManaged(&a, N*sizeof(float));
  cudaMallocManaged(&b, N*sizeof(float));
  cudaMallocManaged(&c, N*sizeof(float));

  for (size_t i = 0; i < N; i++) {
    a[i] = 1.0;
    b[i] = 2.0;
  }

  add<<<256, 256>>>(a, b, c);
  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  for (size_t i = 0; i < N; i++) {
    CHECK(c[i] == 3.0f);
  }

  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
}

struct alloc_cuda_managed {
  size_t N_;
  alloc_cuda_managed(size_t N) : N_(N) {}

  template <typename T>
  void operator()(T& x) const {
    typedef typename std::remove_reference<decltype(*x)>::type x_type;
    // void* p = aligned_malloc(max_num * sizeof(x_type), alignment);
    void* p;
    cudaMallocManaged(&p, N_*sizeof(x_type));
    x = reinterpret_cast<typename std::remove_reference<decltype(x)>::type>(p);
  }
};

struct free_cuda {
  template <typename x_type>
  void operator()(x_type& x) const {
    if (x != nullptr) {
      cudaFree(x);
      x = nullptr;
    }
  }
};

TEST_CASE("Boost fusion stuff", "[Cuda]") {
  size_t N = 256 * 256;

  particle_data data;

  boost::fusion::for_each(data, alloc_cuda_managed(N));

  for (size_t i = 0; i < N; i++) {
    data.dx1[i] = 1.0;
    data.p1[i] = 2.0;
  }

  add<<<256, 256>>>(data.dx1, data.p1, data.x1);
  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  for (size_t i = 0; i < N; i++) {
    CHECK(data.x1[i] == 3.0f);
  }

  boost::fusion::for_each(data, free_cuda());

}
