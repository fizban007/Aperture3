#include "data/particle_data.h"
#include <cuda.h>
#include "catch.hpp"
#include "sim_environment.h"
#include "cuda/constant_mem.h"
#include "cuda/cudaUtility.h"

using namespace Aperture;

struct Test {
  Scalar v[3] = { 0.0 };
};

__constant__ Test dev_tt;
__constant__ SimParamsBase dev_test_params;

__global__
void add(const Scalar* a, const Scalar* b, Scalar* c) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  c[i] = a[i] + b[i];
}

__global__
void print_max_steps() {
  printf("%d\n", dev_test_params.max_steps);
}

TEST_CASE("Launching cuda kernel", "[Cuda]") {
  size_t N = 256*256;
  Scalar *a, *b, *c;
  cudaMallocManaged(&a, N*sizeof(Scalar));
  cudaMallocManaged(&b, N*sizeof(Scalar));
  cudaMallocManaged(&c, N*sizeof(Scalar));

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
    data.x1[i] = 1.0;
    data.p1[i] = 2.0;
  }

  // add<<<256, 256>>>(data.dx1, data.p1, data.x1);
  // Wait for GPU to finish before accessing on host
  // cudaDeviceSynchronize();

  // for (size_t i = 0; i < N; i++) {
  //   CHECK(data.x1[i] == 3.0f);
  // }

  boost::fusion::for_each(data, free_cuda());

}

TEST_CASE("Accessing sim params in kernel", "[Cuda]") {
  SimParams params;
  params.max_steps = 10;
  SimParamsBase* h_params = &params;

  cudaMemcpyToSymbol(dev_test_params, (void*)h_params, sizeof(SimParamsBase));
  CudaCheckError();

  print_max_steps<<<1, 1>>>();
  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
}

TEST_CASE("Transferring and access constant memory", "[Cuda]") {
  Test t1, t2;
  t1.v[0] = 1.0;
  t1.v[1] = 2.0;
  t1.v[2] = 3.0;

  cudaMemcpyToSymbol(dev_tt, (void*)&t1, sizeof(Test));
  cudaMemcpyFromSymbol((void*)&t2, dev_tt, sizeof(Test));

  CHECK(t2.v[0] == 1.0);
  CHECK(t2.v[1] == 2.0);
  CHECK(t2.v[2] == 3.0);
}