#ifndef _ARRAY_H_
#define _ARRAY_H_

#include <algorithm>
#include <cassert>
#include <type_traits>

#include "cuda_runtime.h"

namespace Aperture {

/// This is a simple 1D array wrapping a cuda device pointer
template <typename T>
class cu_array {
 public:
  typedef T data_type;
  typedef T* ptr_type;
  typedef cu_array<T> self_type;

  cu_array();

  explicit cu_array(size_t length, int devId = 0);
  // cu_array(const self_type& other);
  cu_array(self_type&& other);

  ~cu_array();

  /// Assignment operators for copying
  self_type& operator=(const self_type& other);

  /// Move assignment operator
  self_type& operator=(self_type&& other);

  /// Linearized indexing operator, read only
  const data_type& operator[](size_t idx) const {
    assert(idx < m_length);
    return m_data_h[idx];
  }

  /// Linearized indexing operator, read and write
  data_type& operator[](size_t idx) {
    assert(idx < m_length);
    return m_data_h[idx];
  }

  /// Allocate memory on both host and device
  void alloc_mem(size_t N, int deviceId = 0);

  /// Free memory on both host and device
  void free_mem();

  /// Sync the content between host and device
  void copy_to_device(int devId);

  /// Sync the content between host and device
  void copy_to_device();

  /// Sync the content between host and device
  void copy_to_host();

  /// Set the whole array to a single initial value on the host
  void assign(const data_type& value);

  /// Set the whole array to a single initial value through device
  /// kernel
  void assign_dev(const data_type& value);

  /// Set part of the array to a single initial value on the host
  void assign(const data_type& value, size_t num);

  /// Set part of the array to a single initial value through device
  /// kernel
  void assign_dev(const data_type& value, size_t num);

  /// Resize the array.
  void resize(size_t length, int deviceId = 0);

  size_t length() const { return m_length; }
  size_t size() const { return m_length; }
  ptr_type host_ptr() { return m_data_h; }
  const T* host_ptr() const { return m_data_h; }
  ptr_type dev_ptr() { return m_data_d; }
  const T* dev_ptr() const { return m_data_d; }
  int devId() const { return m_devId; }

 private:
  ptr_type m_data_d = nullptr;
  ptr_type m_data_h = nullptr;

  size_t m_length = 0;
  int m_devId = 0;
};

}  // namespace Aperture

#endif  // _ARRAY_H_
