#ifndef _CU_MULTI_ARRAY_H_
#define _CU_MULTI_ARRAY_H_

#include "core/multi_array.h"
#include "core/vec3.h"
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
class cu_multi_array : public multi_array<T> {
 public:
  typedef multi_array<T> base_class;
  typedef T data_type;
  typedef T* ptr_type;
  typedef cu_multi_array<T> self_type;

  /// Default constructor, initializes `_size` to zero and `_data` to
  /// `nullptr`.
  cu_multi_array();

  /// Main constructor, initializes with given width, height, and
  /// depth of the array. Allocate memory in the initialization.
  explicit cu_multi_array(int width, int height = 1, int depth = 1);

  /// Alternative main constructor, takes in an @ref Extent object and
  /// initializes an array of the corresponding extent.
  explicit cu_multi_array(const Extent& extent);

  /// Standard copy constructor.
  // cu_multi_array(const self_type& other);

  /// Standard move constructor.
  cu_multi_array(self_type&& other);

  /// Destructor. Delete the member data array.
  virtual ~cu_multi_array();

  /// Assignment operators for copying
  self_type& operator=(const self_type& other);

  /// Move assignment operator
  self_type& operator=(self_type&& other);

  // /// Linearized indexing operator, read only
  // const data_type& operator[](int idx) const {
  //   assert(idx >= 0 && idx < _size);
  //   return _data_h[idx];
  // }

  // /// Linearized indexing operator, read and write
  // data_type& operator[](int idx) {
  //   assert(idx >= 0 && idx < _size);
  //   return _data_h[idx];
  // }

  // /// Vector indexing operator, read only
  // const data_type& operator()(int x, int y = 0, int z = 0) const {
  //   int idx = x + y * _extent.width() +
  //             z * _extent.width() * _extent.height();
  //   return _data_h[idx];
  // }

  // /// Vector indexing operator, read and write
  // data_type& operator()(int x, int y = 0, int z = 0) {
  //   int idx = x + y * _extent.width() +
  //             z * _extent.width() * _extent.height();
  //   return _data_h[idx];
  // }

  // /// Vector indexing operator using an @ref Index object, read only
  // const data_type& operator()(const Index& index) const {
  //   int idx = index.index(_extent);
  //   return _data_h[idx];
  // }

  // /// Vector indexing operator using an @ref Index object, read and
  // /// write
  // data_type& operator()(const Index& index) {
  //   int idx = index.index(_extent);
  //   return _data_h[idx];
  // }

  /// Copying the entire content from another vector
  void copy_from(const self_type& other);

  /// Set the whole array to a single initial value
  void assign(const data_type& value);

  /// Set the whole array to a single initial value through device
  /// kernel
  void assign_dev(const data_type& value);

  /// Resize the array.
  void resize(int width, int height = 1, int depth = 1);

  /// Resize the array according to an \ref Extent object.
  void resize(Extent extent);

  /// Add a portion of another multi_array to this one
  void add_from(const cu_multi_array<T>& src, Index src_pos, Index pos,
                Extent ext);

  /// Sync the content between host and device
  // void sync_to_device(int devId);

  /// Sync the content between host and device
  void sync_to_device();

  /// Sync the content between host and device
  void sync_to_host();

  /// Allocate memory on both host and device
  void alloc_mem(const Extent& ext);

  /// Free memory on both host and device
  void free_mem();

  /// Direct access to the encapsulated pointer
  // T* data() { return _data_h; }
  // const T* data() const { return _data_h; }
  // cudaPitchedPtr data_d() { return _data_d; }
  cudaPitchedPtr& data_d() { return _data_d; }
  const cudaPitchedPtr& data_d() const { return _data_d; }
  int devId() const { return _devId; }

 private:
  cudaPitchedPtr _data_d;  ///< Pointer to the data stored on the GPU
  int _devId = 0;  ///< Id of the GPU where the memory is allocated

};  // ----- end of class multi_array -----

}  // namespace Aperture

// #include "core/detail/multi_array_impl.hpp"
// #include "core/detail/multi_array_utils.hpp"

#endif  // ----- #ifndef _CU_MULTI_ARRAY_H_  -----
