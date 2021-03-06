#ifndef _CUDA_MEMORY_H_
#define _CUDA_MEMORY_H_

#include "cuda/cuda_control.h"
#include "utils/logger.h"
#include "visit_struct/visit_struct.hpp"

namespace Aperture {

// Helper for allocating Cuda memory

struct alloc_cuda_managed {
  size_t N_;
  alloc_cuda_managed(size_t N) : N_(N) {}

  template <typename T>
  void operator()(const char* name, T& x) const {
    typedef typename std::remove_reference<decltype(*x)>::type x_type;
    void* p;
    // Logger::print_info("--- Allocating {} managed bytes", N_ * sizeof(x_type));
    CudaSafeCall(cudaMallocManaged(&p, N_ * sizeof(x_type)));
    CudaSafeCall(cudaMemAdvise(p, N_ * sizeof(x_type),
                               cudaMemAdviseSetPreferredLocation, 0));
    x = reinterpret_cast<
        typename std::remove_reference<decltype(x)>::type>(p);
  }

  template <typename T>
  void operator()(T& x) const {
    this->operator()("", x);
  }
};

struct alloc_cuda_device {
  size_t N_;
  alloc_cuda_device(size_t N) : N_(N) {}

  template <typename T>
  void operator()(const char* name, T& x) const {
    typedef typename std::remove_reference<decltype(*x)>::type x_type;
    void* p;
    // Logger::print_info("--- Allocating {} bytes", N_ * sizeof(x_type));
    CudaSafeCall(cudaMalloc(&p, N_ * sizeof(x_type)));
    x = reinterpret_cast<
        typename std::remove_reference<decltype(x)>::type>(p);
  }

  template <typename T>
  void operator()(T& x) const {
    this->operator()("", x);
  }
};

struct free_cuda {
  template <typename x_type>
  void operator()(const char* name, x_type& x) const {
    if (x != nullptr) {
      cudaFree(x);
      x = nullptr;
    }
  }

  template <typename T>
  void operator()(T& x) const {
    this->operator()("", x);
  }
};

template <typename StructOfArrays>
void
alloc_struct_of_arrays(StructOfArrays& data, std::size_t max_num) {
  // visit_struct::for_each(data, alloc_cuda_managed(max_num));
  visit_struct::for_each(data, alloc_cuda_device(max_num));
}

template <typename StructOfArrays>
void
alloc_struct_of_arrays_managed(StructOfArrays& data, std::size_t max_num) {
  visit_struct::for_each(data, alloc_cuda_managed(max_num));
  // visit_struct::for_each(data, alloc_cuda_device(max_num));
}

template <typename StructOfArrays>
void
free_struct_of_arrays(StructOfArrays& data) {
  visit_struct::for_each(data, free_cuda());
}

} // namespace Aperture

#endif  // _CUDA_MEMORY_H_
