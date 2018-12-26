#ifndef _MULTIARRAY_H_
#define _MULTIARRAY_H_

#include "data/vec3.h"
#include <algorithm>
#include <cassert>
#include <type_traits>

namespace Aperture {

/// The multi_array class is a unified interface for 1D, 2D and 3D
/// arrays with proper index access and memory management. The most
/// benefit is indexing convenience. One-liners are just implemented
/// in the definition. Other functions are implemented in the followed
/// header file.
template <typename T>
class multi_array {
 public:
  typedef T data_type;
  typedef T* ptr_type;
  typedef multi_array<T> self_type;

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
  multi_array();

  /// Main constructor, initializes with given width, height, and
  /// depth of the array. Allocate memory in the initialization.
  explicit multi_array(int width, int height = 1, int depth = 1);

  /// Alternative main constructor, takes in an @ref Extent object and
  /// initializes an array of the corresponding extent.
  explicit multi_array(const Extent& extent);

  /// Standard copy constructor.
  multi_array(const self_type& other);

  /// Standard move constructor.
  multi_array(self_type&& other);

  /// Destructor. Delete the member data array.
  ~multi_array();

  /// Assignment operators for copying
  self_type& operator=(const self_type& other);

  /// Move assignment operator
  self_type& operator=(self_type&& other);

  /// Vector indexing operator, read only
  const data_type& operator()(int x, int y = 0, int z = 0) const {
    size_t offset = x + y * _pitch +
              z * _pitch * _extent.height();
    return *((ptr_type)((char*)_data + offset));
  }

  /// Vector indexing operator, read and write
  data_type& operator()(int x, int y = 0, int z = 0) {
    size_t offset = x + y * _pitch +
              z * _pitch * _extent.height();
    return *((ptr_type)((char*)_data + offset));
  }

  /// Vector indexing operator using an @ref Index object, read only
  const data_type& operator()(const Index& index) const {
    size_t offset = index.x + index.y * _pitch +
              index.z * _pitch * _extent.height();
    return *((ptr_type)((char*)_data + offset));
  }

  /// Vector indexing operator using an @ref Index object, read and
  /// write
  data_type& operator()(const Index& index) {
    size_t offset = index.x + index.y * _pitch +
              index.z * _pitch * _extent.height();
    return *((ptr_type)((char*)_data + offset));
  }

  /// Linearized indexing operator, read only
  const data_type& operator[](int idx) const {
    int x = idx % _extent.width();
    int y = (idx / _extent.width()) % _extent.height();
    int z = (idx / (_extent.width() * _extent.height()));
    return operator()(x, y, z);
  }

  /// Linearized indexing operator, read and write
  data_type& operator[](int idx) {
    int x = idx % _extent.width();
    int y = (idx / _extent.width()) % _extent.height();
    int z = (idx / (_extent.width() * _extent.height()));
    return operator()(x, y, z);
  }

  /// Copying the entire content from another vector
  void copy_from(const self_type& other);

  /// Set the whole array to a single initial value
  void assign(const data_type& value);

  /// Resize the array.
  void resize(int width, int height = 1, int depth = 1);

  /// Resize the array according to an \ref Extent object.
  void resize(Extent extent);

  /// Allocate memory in an aligned fashion
  void alloc_mem(const Extent& ext);

  /// Free aligned memory
  void free_mem();

  /// Get the dimensions of this array
  /// @return Dimension of the multi-array
  int dim() const { return _dim; }

  // Returns various sizes of the array
  int width() const { return _extent.width(); }
  int height() const { return _extent.height(); }
  int depth() const { return _extent.depth(); }
  size_t pitch() const { return _pitch; }
  size_t size() const { return _size; }
  const Extent& extent() const { return _extent; }

  /// Direct access to the encapsulated pointer
  T* data() { return _data; }
  const T* data() const { return _data; }

 private:
  void find_dim();

  ptr_type _data;        ///< Pointer to the data stored on the host

  Extent _extent;  ///< Extent of the array in all dimensions
  size_t _size;       ///< Total size of the array
  int _dim;        ///< Dimension of the array
  size_t _pitch;      ///< Pitch of the first dimension, in # of bytes

};  // ----- end of class multi_array -----

}  // namespace Aperture

// #include "data/detail/multi_array_impl.hpp"
// #include "data/detail/multi_array_utils.hpp"

#endif  // ----- #ifndef _MULTIARRAY_H_  -----
