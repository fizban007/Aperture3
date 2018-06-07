#ifndef _UTILS_MEMORY_H_
#define _UTILS_MEMORY_H_

#include <cstddef>
#include "boost/container/vector.hpp"
#include "boost/fusion/include/for_each.hpp"
#include "boost/fusion/include/size.hpp"
#include "boost/fusion/include/zip_view.hpp"
#include "cuda/cuda_control.h"

namespace Aperture {

// Helper for allocating Cuda memory

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

template <typename StructOfArrays>
void
alloc_struct_of_arrays(StructOfArrays& data, std::size_t max_num) {
  boost::fusion::for_each(data, alloc_cuda_managed(max_num));
}

template <typename StructOfArrays>
void
free_struct_of_arrays(StructOfArrays& data) {
  boost::fusion::for_each(data, free_cuda());
}

}

#endif  // _UTILS_MEMORY_H_
