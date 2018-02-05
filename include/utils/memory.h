#ifndef _UTILS_MEMORY_H_
#define _UTILS_MEMORY_H_

#include <cstddef>
#include "boost/container/vector.hpp"
#include "boost/fusion/include/for_each.hpp"
#include "boost/fusion/include/size.hpp"
#include "boost/fusion/include/zip_view.hpp"


namespace Aperture {

// Helper for allocating memory with a given alignment requirement

/// Malloc an aligned memory region of size size, with specified alignment. It
/// is required that alignment is smaller than 0x8000. Note: This function does
/// not initialize the new allocated memory. Need to call initialize by hand
/// afterwards
void* aligned_malloc(std::size_t size, std::size_t alignment);
void aligned_free(void* p);

template <typename StructOfArrays>
void
alloc_struct_of_arrays(StructOfArrays& data, std::size_t max_num, std::size_t alignment) {
  boost::fusion::for_each(data, [max_num, alignment](auto& x) {
      typedef typename std::remove_reference<decltype(*x)>::type x_type;
      void* p = aligned_malloc(max_num * sizeof(x_type), alignment);
      x = reinterpret_cast<typename std::remove_reference<decltype(x)>::type>(p);
    });
}

template <typename StructOfArrays>
void
free_struct_of_arrays(StructOfArrays& data) {
  boost::fusion::for_each(data, [](auto& x) {
      // x = nullptr;
      if (x != nullptr) {
        aligned_free(reinterpret_cast<void*>(x));
        x = nullptr;
      }
    });
}

}

#endif  // _UTILS_MEMORY_H_
