#ifndef _ARRAY_H_
#define _ARRAY_H_

#include <algorithm>
#include <cassert>

namespace Aperture {

/// This is a simple 1D array wrapping a cuda device pointer
template <typename T>
class array {
 public:
  typedef T data_type;
  typedef T* ptr_type;
  typedef array<T> self_type;

  array();

  explicit array(size_t size);
  array(const self_type& other);
  array(self_type&& other);

  ~array();

  /// Assignment operators for copying
  self_type& operator=(const self_type& other);

  /// Move assignment operator
  self_type& operator=(self_type&& other);

  /// Linearized indexing operator, read only
  const data_type& operator[](size_t idx) const {
    assert(idx < m_size);
    return m_data_h[idx];
  }

  /// Linearized indexing operator, read and write
  data_type& operator[](size_t idx) {
    assert(idx < m_size);
    return m_data_h[idx];
  }

  /// Allocate memory on both host and device
  void alloc_mem(size_t N);

  /// Free memory on both host and device
  void free_mem();

  /// Sync the content between host and device
  void sync_to_device();

  /// Sync the content between host and device
  void sync_to_host();

  /// Set the whole array to a single initial value on the host
  void assign(const data_type& value);

  /// Set part of the array to a single initial value on the host
  void assign(const data_type& value, size_t num);

  /// Set the whole array to a single initial value on the host
  void assign_dev(const data_type& value);

  /// Set part of the array to a single initial value on the host
  void assign_dev(const data_type& value, size_t num);

  /// Copy from another array
  void copy_from(const self_type& other);

  /// Resize the array.
  void resize(size_t size);

  size_t size() const { return m_size; }
  ptr_type host_ptr() { return m_data_h; }
  const T* host_ptr() const { return m_data_h; }
  ptr_type dev_ptr() { return m_data_d; }
  const T* dev_ptr() const { return m_data_d; }

 private:
  ptr_type m_data_d = nullptr;
  ptr_type m_data_h = nullptr;

  size_t m_size = 0;
};

}  // namespace Aperture

#endif  // _ARRAY_H_
