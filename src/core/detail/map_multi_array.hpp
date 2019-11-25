#ifndef _MAP_MULTI_ARRAY_H_
#define _MAP_MULTI_ARRAY_H_

#include "core/detail/op.hpp"
#include "core/multi_array.h"
#include "omp.h"

namespace Aperture {

namespace detail {

////////////////////////////////////////////////////////////////////////////////
///  Mapping an operation over a multiarray.
////////////////////////////////////////////////////////////////////////////////
template <typename T, typename UnaryOp>
void
map_multi_array(multi_array<T>& array, const Index& start,
                const Extent& range, UnaryOp op) {
  for (int k = start.z; k < range.depth(); ++k) {
    size_t k_offset = k * array.height();
    for (int j = start.y; j < range.height(); ++j) {
      size_t offset = (j + k_offset) * array.width();
#pragma omp simd
      for (int i = start.x; i < range.width(); ++i) {
        // TODO: Add optimization to these routines to exploit
        // parallelization
        op(array[i + offset]);
      }
    }
  }
}

template <typename T, typename UnaryOp>
void
map_multi_array_slow(multi_array<T>& array, const Index& start,
                     const Extent& range, UnaryOp op) {
  for (int k = start.z; k < range.depth(); ++k) {
    size_t k_offset = k * array.width() * array.height();
    for (int j = start.y; j < range.height(); ++j) {
      size_t j_offset = j * array.width();
      for (int i = start.x; i < range.width(); ++i) {
        // TODO: Add optimization to these routines to exploit
        // parallelization
        op(array[i + j_offset + k_offset]);
      }
    }
  }
}
////////////////////////////////////////////////////////////////////////////////
///  Mapping an operation over two multiarrays.
////////////////////////////////////////////////////////////////////////////////
template <typename T, typename BinaryOp>
void
map_multi_array(multi_array<T>& output, const Index& output_start,
                const multi_array<T>& input, const Index& input_start,
                const Extent& range, BinaryOp op) {
  for (int k = 0; k < range.depth(); ++k) {
    size_t k_offset_out = (k + output_start.z) * output.height();
    size_t k_offset_in = (k + input_start.z) * input.height();
    for (int j = 0; j < range.height(); ++j) {
      size_t offset_out =
          (j + output_start.y + k_offset_out) * output.width();
      size_t offset_in =
          (j + input_start.y + k_offset_in) * input.width();
#pragma omp simd
      for (int i = 0; i < range.width(); ++i) {
        // TODO: Add optimization to these routines to exploit
        // parallelization
        op(output[(i + output_start.x) + offset_out],
           input[(i + input_start.x) + offset_in]);
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
///  Mapping an operation over two multiarrays, assuming start index is
///  always zero.
////////////////////////////////////////////////////////////////////////////////
template <typename T, typename BinaryOp>
void
map_multi_array(multi_array<T>& output, const multi_array<T>& input,
                const Extent& range, BinaryOp op) {
  for (int k = 0; k < range.depth(); ++k) {
    size_t k_offset = k * output.height();
    for (int j = 0; j < range.height(); ++j) {
      size_t offset = (j + k_offset) * output.width();
#pragma omp simd
      for (int i = 0; i < range.width(); ++i) {
        // TODO: Add optimization to these routines to exploit
        // parallelization
        op(output[i + offset], input[i + offset]);
      }
    }
  }
}

template <typename T, typename BinaryOp>
void
map_multi_array_slow(multi_array<T>& output, const Index& output_start,
                     const multi_array<T>& input,
                     const Index& input_start, const Extent& range,
                     BinaryOp op) {
  for (int k = 0; k < range.depth(); ++k) {
    size_t k_offset_out =
        (k + output_start.z) * output.width() * output.height();
    size_t k_offset_in =
        (k + input_start.z) * input.width() * input.height();
    for (int j = 0; j < range.height(); ++j) {
      size_t j_offset_out = (j + output_start.y) * output.width();
      size_t j_offset_in = (j + input_start.y) * input.width();
      for (int i = 0; i < range.width(); ++i) {
        // TODO: Add optimization to these routines to exploit
        // parallelization
        op(output[(i + output_start.x) + j_offset_out + k_offset_out],
           input[(i + input_start.x) + j_offset_in + k_offset_in]);
      }
    }
  }
}
}  // namespace detail

}  // namespace Aperture

#endif  // _MAP_MULTI_ARRAY_H_
