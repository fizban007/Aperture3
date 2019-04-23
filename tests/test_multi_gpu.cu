#include <iostream>
#include <vector>

__global__
void set_value(float* array, int size, float value) {
  for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < size;
       i += blockDim.x * gridDim.x) {
    array[i] = value;
  }
}

__global__
void get_value(float* array, int pos) {
  printf("value is %f\n", array[pos]);
}

int main(int argc, char *argv[]) {
  // Poll the system to detect how many GPUs are on the node
  int n_devices;
  cudaGetDeviceCount(&n_devices);
  if (n_devices <= 0) {
    exit(1);
  }

  std::vector<int> dev_map(n_devices);
  // Logger::print_info("Found {} Cuda devices:", n_devices);
  for (int i = 0; i < n_devices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    // Logger::print_info("    Device Number: {}", i);
    // Logger::print_info("    Device Name: {}", prop.name);
    // Logger::print_info("    Device Total Memory: {}MiB",
    //                    prop.totalGlobalMem / (1024 * 1024));
    dev_map[i] = i;
  }

  float *v1, *v2;
  int N = 10000;
  cudaSetDevice(dev_map[0]);
  cudaMallocManaged(&v1, N*sizeof(float));
  set_value<<<64, 128>>>(v1, N, 1.0);
  cudaSetDevice(dev_map[1]);
  cudaMallocManaged(&v2, N*sizeof(float));
  set_value<<<64, 128>>>(v2, N, 2.0);
  cudaDeviceSynchronize();
  get_value<<<1,1>>>(v1, 4000);
  cudaDeviceSynchronize();

  return 0;
}