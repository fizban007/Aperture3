#ifndef  _MULTI_ARRAY_UTILS_HPP_
#define  _MULTI_ARRAY_UTILS_HPP_

// #include "cuda/cuda_control.h"
#include "data/vec3.h"
#include "data/detail/multi_array_iter_impl.hpp"

namespace Aperture {

namespace detail {

////////////////////////////////////////////////////////////////////////////////
///  Mapping an operation over a multiarray.
////////////////////////////////////////////////////////////////////////////////
template <typename InputIterator, typename UnaryOp>
void map_multi_array(const InputIterator& it, const Extent& range, UnaryOp op) {
  for (int k = 0; k < range.depth(); ++k) {
    for (int j = 0; j < range.height(); ++j) {
      for (int i = 0; i < range.width(); ++i) {
        // TODO: Add optimization to these routines to exploit parallelization
        op(it(i, j, k));
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
///  Mapping an operation over two multiarrays.
////////////////////////////////////////////////////////////////////////////////
template <typename InputIterator, typename OutputIterator, typename BinaryOp>
void map_multi_array(const OutputIterator& output, const InputIterator& input,
                     const Extent& range, BinaryOp op) {
  for (int k = 0; k < range.depth(); ++k) {
    for (int j = 0; j < range.height(); ++j) {
      for (int i = 0; i < range.width(); ++i) {
        // TODO: Add optimization to these routines to exploit parallelization
        op(output(i, j, k), input(i, j, k));
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
//  Operator functors
////////////////////////////////////////////////////////////////////////////////
template <typename T>
struct Op_Assign {
  void operator()(T& dest, const T& value) const { dest = value; }
};

template <typename T>
struct Op_PlusAssign {
  void operator()(T& dest, const T& value) const { dest += value; }
};

template <typename T>
struct Op_MinusAssign {
  void operator()(T& dest, const T& value) const { dest -= value; }
};

template <typename T>
struct Op_MultAssign {
  void operator()(T& dest, const T& value) const { dest *= value; }
};

template <typename T>
struct Op_AssignConst {
  T _value;
  Op_AssignConst(const T& value) : _value(value) {}

  void operator()(T& dest) const { dest = _value; }
};

template <typename T>
struct Op_MultConst {
  T _value;
  Op_MultConst(const T& value) : _value(value) {}

  void operator()(T& dest) const { dest *= _value; }
};

template <typename T>
struct Op_PlusConst {
  T _value;
  Op_PlusConst(const T& value) : _value(value) {}

  void operator()(T& dest) const { dest += _value; }
};

template <typename T>
struct Op_MinusConst {
  T _value;
  Op_MinusConst(const T& value) : _value(value) {}

  void operator()(T& dest) const { dest -= _value; }
};

}

template <typename It>
void check_bounds(const It& iterator, const Extent& extent) {
  // Check bounds
  if (iterator + extent - Extent(1, 1, 1) > iterator.array().end()) {
    std::cerr << iterator.pos() << std::endl;
    throw std::invalid_argument("Index out of bounds in array operation!");
  }
}

////////////////////////////////////////////////////////////////////////////////
///  Copy a chunk of data from input to output, with size specified
///  with extent.
////////////////////////////////////////////////////////////////////////////////
template <typename InputIt, typename OutputIt>
void copy(const OutputIt& output, const InputIt& input, const Extent& extent) {
  check_bounds(input, extent);
  check_bounds(output, extent);

  detail::map_multi_array(output, input, extent, detail::Op_Assign<typename InputIt::data_type>());
}

////////////////////////////////////////////////////////////////////////////////
///  Copy a chunk of data from input to output, with size specified
///  with extent, assuming the output is a linearized array.
////////////////////////////////////////////////////////////////////////////////
template <typename InputIt, typename OutputIt>
void copy_to_linear(const OutputIt& output, const InputIt& input, const Extent& extent) {
  check_bounds(input, extent);
  // check_bounds(output, extent);

  // detail::map_multi_array(output, input, extent, Op_Assign<typename InputIt::data_type>());
  for (int k = 0; k < extent.depth(); ++k) {
    for (int j = 0; j < extent.height(); ++j) {
      for (int i = 0; i < extent.width(); ++i) {
        int index = Index(i, j, k).index(extent);
        output[index] = input(i, j, k);
        // op(output(i, j, k), input(i, j, k));
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
///  Copy a chunk of data from input to output, with size specified
///  with extent, assuming the iutput is a linearized array.
////////////////////////////////////////////////////////////////////////////////
template <typename InputIt, typename OutputIt>
void copy_from_linear(const OutputIt& output, const InputIt& input, const Extent& extent) {
  // check_bounds(input, extent);
  check_bounds(output, extent);

  // detail::map_multi_array(output, input, extent, Op_Assign<typename InputIt::data_type>());
  for (int k = 0; k < extent.depth(); ++k) {
    for (int j = 0; j < extent.height(); ++j) {
      for (int i = 0; i < extent.width(); ++i) {
        int index = Index(i, j, k).index(extent);
        output(i, j, k) = input[index];
        // op(output(i, j, k), input(i, j, k));
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
///  Add a chunk of data from input to output, with size specified
///  with extent, assuming the iutput is a linearized array.
////////////////////////////////////////////////////////////////////////////////
template <typename InputIt, typename OutputIt>
void add_from_linear(const OutputIt& output, const InputIt& input, const Extent& extent) {
  // check_bounds(input, extent);
  check_bounds(output, extent);

  // detail::map_multi_array(output, input, extent, Op_Assign<typename InputIt::data_type>());
  for (int k = 0; k < extent.depth(); ++k) {
    for (int j = 0; j < extent.height(); ++j) {
      for (int i = 0; i < extent.width(); ++i) {
        int index = Index(i, j, k).index(extent);
        output(i, j, k) += input[index];
        // op(output(i, j, k), input(i, j, k));
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
///  Fill a chunk of input array with uniform value, the size of the
///  chunk given by extent.
////////////////////////////////////////////////////////////////////////////////
template <typename InputIt, typename T>
void fill(const InputIt& input, const Extent& extent, T value) {
  check_bounds(input, extent);

  detail::map_multi_array(input, extent, detail::Op_AssignConst<typename InputIt::data_type>(value));
}

////////////////////////////////////////////////////////////////////////////////
///  Multiply a chunk of the input array against the output array term
///  by term.
////////////////////////////////////////////////////////////////////////////////
template <typename InputIt, typename OutputIt>
void multiply(const OutputIt& output, const InputIt& input,
              const Extent& extent) {
  check_bounds(input, extent);
  check_bounds(output, extent);

  detail::map_multi_array(output, input, extent, detail::Op_MultAssign<typename InputIt::data_type>());
}

////////////////////////////////////////////////////////////////////////////////
///  Multiply a chunk of the input array by a single value.
////////////////////////////////////////////////////////////////////////////////
template <typename InputIt, typename T>
void multiply(const InputIt& input, const Extent& extent, T value) {
  check_bounds(input, extent);

  detail::map_multi_array(input, extent, detail::Op_MultConst<typename InputIt::data_type>(value));
}

////////////////////////////////////////////////////////////////////////////////
///  Add a chunk of the input array to the output array, size of the
///  chunk given by extent.
////////////////////////////////////////////////////////////////////////////////
template <typename InputIt, typename OutputIt>
void add(const OutputIt& output, const InputIt& input, const Extent& extent) {
  check_bounds(input, extent);
  check_bounds(output, extent);

  detail::map_multi_array(output, input, extent, detail::Op_PlusAssign<typename InputIt::data_type>());
}

////////////////////////////////////////////////////////////////////////////////
///  Add a chunk of the input array by a single value.
////////////////////////////////////////////////////////////////////////////////
template <typename InputIt, typename T>
void add(const InputIt& input, const Extent& extent, T value) {
  check_bounds(input, extent);

  detail::map_multi_array(input, extent, detail::Op_PlusConst<typename InputIt::data_type>(value));
}

////////////////////////////////////////////////////////////////////////////////
///  Subtract a chunk of the input array from the output array, size
///  of the chunk given by extent.
////////////////////////////////////////////////////////////////////////////////
template <typename InputIt, typename OutputIt>
void subtract(const OutputIt& output, const InputIt& input, const Extent& extent) {
  check_bounds(input, extent);
  check_bounds(output, extent);

  detail::map_multi_array(output, input, extent, detail::Op_MinusAssign<typename InputIt::data_type>());
}

////////////////////////////////////////////////////////////////////////////////
///  Subtract a chunk of the input array by a single value.
////////////////////////////////////////////////////////////////////////////////
template <typename InputIt, typename T>
void subtract(const InputIt& input, const Extent& extent, T value) {
  check_bounds(input, extent);

  detail::map_multi_array(input, extent, detail::Op_MinusConst<typename InputIt::data_type>(value));
}


}

#endif   // _MULTI_ARRAY_UTILS_HPP_
