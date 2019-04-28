#include "catch.hpp"
#include "cuda/cudaUtility.h"
#include "cuda/utils/typed_pitchedptr.cuh"
#include <vector>

using namespace Aperture;

template <typename T>
__global__ void assign_values(typed_pitchedptr<T> ptr) {
  ptr(threadIdx.x, blockIdx.x) = threadIdx.x * blockIdx.x;
}

TEST_CASE("Simple usage", "[pitchedptr]") {
  cudaPitchedPtr p2f, p3f;
  cudaPitchedPtr p2d, p3d;

  CudaSafeCall(cudaMalloc3D(&p2f, make_cudaExtent(100 * sizeof(float), 100, 1)));
  CudaSafeCall(cudaMalloc3D(&p2d, make_cudaExtent(100 * sizeof(double), 100, 1)));
  CudaSafeCall(cudaMalloc3D(&p3f, make_cudaExtent(100 * sizeof(float), 100, 100)));
  CudaSafeCall(cudaMalloc3D(&p3d, make_cudaExtent(100 * sizeof(double), 100, 100)));

  std::vector<float> v2f(100*100), v3f(100*100*100);
  std::vector<double> v2d(100*100), v3d(100*100*100);

  assign_values<<<100, 100>>>(typed_pitchedptr<float>(p2f));
  assign_values<<<100, 100>>>(typed_pitchedptr<double>(p2d));
  CudaSafeCall(cudaDeviceSynchronize());

  cudaMemcpy3DParms pv2f = {0};
  cudaMemcpy3DParms pv2d = {0};
  pv2f.srcPtr = p2f;
  pv2f.dstPtr = make_cudaPitchedPtr(v2f.data(), 100*sizeof(float), 100, 100);
  pv2f.srcPos = make_cudaPos(0, 0, 0);
  pv2f.dstPos = make_cudaPos(0, 0, 0);
  pv2f.extent = make_cudaExtent(100*sizeof(float), 100, 1);
  pv2f.kind = cudaMemcpyDeviceToHost;
  pv2d.srcPtr = p2d;
  pv2d.dstPtr = make_cudaPitchedPtr(v2d.data(), 100*sizeof(double), 100, 100);
  pv2d.srcPos = make_cudaPos(0, 0, 0);
  pv2d.dstPos = make_cudaPos(0, 0, 0);
  pv2d.extent = make_cudaExtent(100*sizeof(double), 100, 1);
  pv2d.kind = cudaMemcpyDeviceToHost;
  CudaSafeCall(cudaMemcpy3D(&pv2f));
  CudaSafeCall(cudaMemcpy3D(&pv2d));

  for(int i = 0; i < 100; i++) {
    for (int j = 0; j < 100; j++) {
      CHECK(v2f[i + j * 100] == Approx(i * j));
      CHECK(v2d[i + j * 100] == Approx(i * j));
    }
  }

  CudaSafeCall(cudaFree(p2f.ptr));
  CudaSafeCall(cudaFree(p2d.ptr));
  CudaSafeCall(cudaFree(p3f.ptr));
  CudaSafeCall(cudaFree(p3d.ptr));
}
