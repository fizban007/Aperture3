#include "catch.hpp"
#include "cuda/cudaUtility.h"
#include "cuda/utils/pitchptr.cuh"
#include <vector>

using namespace Aperture;

template <typename T>
__global__ void
assign_values(pitchptr<T> ptr) {
  ptr(threadIdx.x, blockIdx.x) = threadIdx.x * blockIdx.x;
}

template <typename T>
__global__ void
assign_values3d(pitchptr<T> ptr) {
  ptr(threadIdx.x, threadIdx.y, blockIdx.x) =
      threadIdx.x * blockIdx.x * threadIdx.y;
}

TEST_CASE("Simple usage", "[pitchedptr]") {
  cudaPitchedPtr p2f, p3f;
  cudaPitchedPtr p2d, p3d;

  CudaSafeCall(
      cudaMalloc3D(&p2f, make_cudaExtent(100 * sizeof(float), 100, 1)));
  CudaSafeCall(cudaMalloc3D(
      &p2d, make_cudaExtent(100 * sizeof(double), 100, 1)));
  CudaSafeCall(
      cudaMalloc3D(&p3f, make_cudaExtent(8 * sizeof(float), 8, 8)));
  CudaSafeCall(
      cudaMalloc3D(&p3d, make_cudaExtent(8 * sizeof(double), 8, 8)));

  std::vector<float> v2f(100 * 100), v3f(8 * 8 * 8);
  std::vector<double> v2d(100 * 100), v3d(8 * 8 * 8);

  assign_values<<<100, 100>>>(pitchptr<float>(p2f));
  assign_values<<<100, 100>>>(pitchptr<double>(p2d));
  assign_values3d<<<8, dim3(8, 8)>>>(pitchptr<float>(p3f));
  assign_values3d<<<8, dim3(8, 8)>>>(pitchptr<double>(p3d));
  CudaSafeCall(cudaDeviceSynchronize());

  cudaMemcpy3DParms pv2f = {0};
  cudaMemcpy3DParms pv2d = {0};
  pv2f.srcPtr = p2f;
  pv2f.dstPtr =
      make_cudaPitchedPtr(v2f.data(), 100 * sizeof(float), 100, 100);
  pv2f.srcPos = make_cudaPos(0, 0, 0);
  pv2f.dstPos = make_cudaPos(0, 0, 0);
  pv2f.extent = make_cudaExtent(100 * sizeof(float), 100, 1);
  pv2f.kind = cudaMemcpyDeviceToHost;
  pv2d.srcPtr = p2d;
  pv2d.dstPtr =
      make_cudaPitchedPtr(v2d.data(), 100 * sizeof(double), 100, 100);
  pv2d.srcPos = make_cudaPos(0, 0, 0);
  pv2d.dstPos = make_cudaPos(0, 0, 0);
  pv2d.extent = make_cudaExtent(100 * sizeof(double), 100, 1);
  pv2d.kind = cudaMemcpyDeviceToHost;
  CudaSafeCall(cudaMemcpy3D(&pv2f));
  CudaSafeCall(cudaMemcpy3D(&pv2d));

  for (int i = 0; i < 100; i++) {
    for (int j = 0; j < 100; j++) {
      CHECK(v2f[i + j * 100] == Approx(i * j));
      CHECK(v2d[i + j * 100] == Approx(i * j));
    }
  }

  cudaMemcpy3DParms pv3f = {0};
  cudaMemcpy3DParms pv3d = {0};
  pv3f.srcPtr = p3f;
  pv3f.dstPtr =
      make_cudaPitchedPtr(v3f.data(), 8 * sizeof(float), 8, 8);
  pv3f.srcPos = make_cudaPos(0, 0, 0);
  pv3f.dstPos = make_cudaPos(0, 0, 0);
  pv3f.extent = make_cudaExtent(8 * sizeof(float), 8, 8);
  pv3f.kind = cudaMemcpyDeviceToHost;
  pv3d.srcPtr = p3d;
  pv3d.dstPtr =
      make_cudaPitchedPtr(v3d.data(), 8 * sizeof(double), 8, 8);
  pv3d.srcPos = make_cudaPos(0, 0, 0);
  pv3d.dstPos = make_cudaPos(0, 0, 0);
  pv3d.extent = make_cudaExtent(8 * sizeof(double), 8, 8);
  pv3d.kind = cudaMemcpyDeviceToHost;
  CudaSafeCall(cudaMemcpy3D(&pv3f));
  CudaSafeCall(cudaMemcpy3D(&pv3d));

  for (int k = 0; k < 8; k++) {
    for (int j = 0; j < 8; j++) {
      for (int i = 0; i < 8; i++) {
        CHECK(v3f[i + j * 8 + k * 64] == Approx(i * j * k));
        CHECK(v3d[i + j * 8 + k * 64] == Approx(i * j * k));
      }
    }
  }

  CudaSafeCall(cudaFree(p2f.ptr));
  CudaSafeCall(cudaFree(p2d.ptr));
  CudaSafeCall(cudaFree(p3f.ptr));
  CudaSafeCall(cudaFree(p3d.ptr));
}
