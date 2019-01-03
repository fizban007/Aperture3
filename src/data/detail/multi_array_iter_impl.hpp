#ifndef _MULTI_ARRAY_ITER_IMPL_H_
#define _MULTI_ARRAY_ITER_IMPL_H_

#include "data/cu_multi_array.h"

namespace Aperture {

template <typename T>
template <bool isConst>
class cu_multi_array<T>::const_nonconst_iterator {
 public:
  // Typedefs for easier writing and const/nonconst switch
  typedef cu_multi_array<T> array_type;
  typedef const_nonconst_iterator<isConst> self_type;
  typedef Index index_type;
  typedef T data_type;
  typedef
      typename std::conditional<isConst, const T*, T*>::type ptr_type;
  typedef
      typename std::conditional<isConst, const T&, T&>::type ref_type;
  typedef typename std::conditional<isConst, const array_type&,
                                    array_type&>::type arr_ref_type;
  typedef typename std::conditional<isConst, const array_type*,
                                    array_type*>::type arr_ptr_type;

  // Constructors and destructors
  const_nonconst_iterator(arr_ref_type array, index_type pos)
      : _array(array), _pos(pos) {}

  const_nonconst_iterator(arr_ref_type array, int x, int y, int z)
      : _array(array), _pos(x, y, z) {}

  const_nonconst_iterator(arr_ref_type array, int idx)
      : _array(array), _pos(idx, _array.extent()) {
    // std::cout << "Index is " << idx << std::endl;
  }

  /// Copy constructor.
  const_nonconst_iterator(const self_type& iter)
      : _array(iter._array), _pos(iter._pos) {}

  ~const_nonconst_iterator() {}

  /// Add-assign operator, takes in 3 integers
  self_type& operator+=(const Vec3<int>& extent) {
    _pos.x += extent.x;
    _pos.y += extent.y;
    _pos.z += extent.z;
    return (*this);
  }

  /// Minus-assign operator, takes in 3 integers
  self_type& operator-=(const Vec3<int>& extent) {
    _pos.x -= extent.x;
    _pos.y -= extent.y;
    _pos.z -= extent.z;
    return (*this);
  }

  /// Add operator, takes in 3 integers
  self_type operator+(const Vec3<int>& extent) const {
    self_type tmp(*this);
    tmp += extent;
    return tmp;
  }

  /// Minus operator, takes in 3 integers
  self_type operator-(const Vec3<int>& extent) const {
    self_type tmp(*this);
    tmp -= extent;
    return tmp;
  }

  /// Comparison operator, greater than. Iterators should be able to
  /// compare to each other regardless of constness.
  template <bool Const>
  bool operator>(const const_nonconst_iterator<Const>& iter) const {
    return (_pos.index(_array.extent()) >
            iter.pos().index(iter.array().extent()));
  }

  /// Comparison operator, less than. Iterators should be able to
  /// compare to each other regardless of constness.
  template <bool Const>
  bool operator<(const const_nonconst_iterator<Const>& iter) const {
    return (_pos.index(_array.extent()) <
            iter.pos().index(iter.array().extent()));
  }

  ////////////////////////////////////////////////////////////////////////////////
  //  Access operators
  ////////////////////////////////////////////////////////////////////////////////

  /// Pointer dereference, returns regular or const reference
  /// depending on whether the iterator is const
  ref_type operator*() const { return _array(_pos); }

  /// Index operator, returns regular or const reference
  /// depending on whether the iterator is const
  ref_type operator()(int x, int y = 0, int z = 0) const {
    return _array(_pos.x + x, _pos.y + y, _pos.z + z);
  }

  /// Index operator, returns regular or const reference
  /// depending on whether the iterator is const
  ref_type operator()(const Index& idx) const {
    return operator()(idx.x, idx.y, idx.z);
  }

  /// Linear index operator, returns regular or const reference
  /// depending on whether the iterator is const
  ref_type operator[](int n) const {
    return _array[_pos.index(_array.extent()) + n];
  }

  /// Returns a const reference to the array of this iterator.
  const array_type& array() const { return _array; }

  /// Returns a const reference to the position that this iterator
  /// points to.
  const index_type& pos() const { return _pos; }

 private:
  arr_ref_type _array;  ///< A reference to the underlying array
  index_type _pos;      ///< Position that this iterator points to
};                      // ----- end of class const_nonconst_iterator

}  // namespace Aperture

#endif  // _MULTI_ARRAY_ITER_IMPL_H_
