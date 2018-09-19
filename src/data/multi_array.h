#ifndef _MULTIARRAY_H_
#define _MULTIARRAY_H_

#include "data/vec3.h"
#include <algorithm>
#include <cassert>
#include <type_traits>

#include "cuda_runtime.h"

namespace Aperture {

/// The multi_array class is a unified interface for 1D, 2D and 3D
/// arrays with proper index access and memory management. The most
/// benefit is indexing convenience. One-liners are just implemented
/// in the definition. Other functions are implemented in the followed
/// header file.
template <typename T>
class MultiArray {
 public:
  typedef T data_type;
  typedef T* ptr_type;
  typedef MultiArray<T> self_type;

  /// Iterator for the multi_array class, defines both the const and
  /// nonconst iterators at the same time.
  // template <bool isConst>
  // class const_nonconst_iterator;

  // typedef const_nonconst_iterator<false> iterator;
  // typedef const_nonconst_iterator<true> const_iterator;

  ////////////////////////////////////////////////////////////////////////////////
  //  Constructors
  ////////////////////////////////////////////////////////////////////////////////

  /// Default constructor, initializes `_size` to zero and `_data` to
  /// `nullptr`.
  MultiArray();

  /// Main constructor, initializes with given width, height, and
  /// depth of the array. Allocate memory in the initialization.
  explicit MultiArray(int width, int height = 1, int depth = 1,
                      int deviceId = 0);

  /// Alternative main constructor, takes in an @ref Extent object and
  /// initializes an array of the corresponding extent.
  explicit MultiArray(const Extent& extent, int deviceId = 0);

  /// Standard copy constructor.
  MultiArray(const self_type& other);

  /// Standard move constructor.
  MultiArray(self_type&& other);

  /// Destructor. Delete the member data array.
  ~MultiArray();

  /// Assignment operators for copying
  self_type& operator=(const self_type& other);

  /// Move assignment operator
  self_type& operator=(self_type&& other);

  /// Linearized indexing operator, read only
  const data_type& operator[](int idx) const {
    assert(idx >= 0 && idx < _size);
    return _data_h[idx];
  }

  /// Linearized indexing operator, read and write
  data_type& operator[](int idx) {
    assert(idx >= 0 && idx < _size);
    return _data_h[idx];
  }

  /// Vector indexing operator, read only
  const data_type& operator()(int x, int y = 0, int z = 0) const {
    int idx = x + y * _extent.width() +
              z * _extent.width() * _extent.height();
    return _data_h[idx];
  }

  /// Vector indexing operator, read and write
  data_type& operator()(int x, int y = 0, int z = 0) {
    int idx = x + y * _extent.width() +
              z * _extent.width() * _extent.height();
    return _data_h[idx];
  }

  /// Vector indexing operator using an @ref Index object, read only
  const data_type& operator()(const Index& index) const {
    int idx = index.index(_extent);
    return _data_h[idx];
  }

  /// Vector indexing operator using an @ref Index object, read and
  /// write
  data_type& operator()(const Index& index) {
    int idx = index.index(_extent);
    return _data_h[idx];
  }

  /// Copying the entire content from another vector
  void copyFrom(const self_type& other);

  /// Set the whole array to a single initial value
  void assign(const data_type& value);
  // std::cout << "Assigning value " << value << " for size " << _size
  // << std::endl; std::fill_n(_data, _size, value);
  // }
  /// Set the whole array to a single initial value through device
  /// kernel
  void assign_dev(const data_type& value);

  /// Resize the array.
  void resize(int width, int height = 1, int depth = 1,
              int deviceId = 0);

  /// Resize the array according to an \ref Extent object.
  void resize(Extent extent, int deviceId = 0);

  /// Sync the content between host and device
  void sync_to_device(int devId);

  /// Sync the content between host and device
  void sync_to_device();

  /// Sync the content between host and device
  void sync_to_host();

  /// Allocate memory on both host and device
  void alloc_mem(const Extent& ext, int deviceId = 0);

  /// Free memory on both host and device
  void free_mem();

  ////////////////////////////////////////////////////////////////////////////////
  //  Iterators
  ////////////////////////////////////////////////////////////////////////////////

  // iterator begin() { return iterator(*this, 0); }
  // iterator end() { return iterator(*this, _size); }
  // iterator index(int x, int y = 0, int z = 0) {
  //   return iterator(*this, x, y, z);
  // }
  // iterator index(const Index& index) {
  //   return iterator(*this, index.x, index.y, index.z);
  // }

  // const_iterator begin() const { return const_iterator(*this, 0); }
  // const_iterator end() const { return const_iterator(*this, _size); }
  // const_iterator index(int x, int y = 0, int z = 0) const {
  //   return const_iterator(*this, x, y, z);
  // }
  // const_iterator index(const Index& index) const {
  //   return const_iterator(*this, index.x, index.y, index.z);
  // }

  /// Get the dimensions of this array
  ///
  /// @return Dimension of the multi-array
  ///
  int dim() const { return _dim; }

  // Returns various sizes of the array
  int width() const { return _extent.width(); }
  int height() const { return _extent.height(); }
  int depth() const { return _extent.depth(); }
  int size() const { return _size; }
  const Extent& extent() const { return _extent; }

  /// Setting data ptr
  // void set_data(T* p) {
  //   if (_data != nullptr)
  //     delete[] _data;
  //   _data = p;
  // }

  /// Direct access to the encapsulated pointer
  T* data() { return _data_h; }
  const T* data() const { return _data_h; }
  cudaPitchedPtr& data_d() { return _data_d; }
  cudaPitchedPtr data_d() const { return _data_d; }
  int devId() const { return _devId; }

 private:
  void find_dim();

  cudaPitchedPtr _data_d;  ///< Pointer to the data stored on the GPU
  ptr_type _data_h;        ///< Pointer to the data stored on the host

  Extent _extent;  ///< Extent of the array in all dimensions
  int _size;       ///< Total size of the array
  int _dim;        ///< Dimension of the array
  int _devId = 0;  ///< Id of the GPU where the memory is allocated

};  // ----- end of class multi_array -----

}  // namespace Aperture

// #include "data/detail/multi_array_impl.hpp"
// #include "data/detail/multi_array_utils.hpp"

#endif  // ----- #ifndef _MULTIARRAY_H_  -----
