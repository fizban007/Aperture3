#ifndef _OP_H_
#define _OP_H_

#include "cuda/cuda_control.h"

namespace Aperture {

namespace detail {

////////////////////////////////////////////////////////////////////////////////
//  Operator functors
////////////////////////////////////////////////////////////////////////////////
template <typename T>
struct Op_Assign {
  HD_INLINE void operator()(T& dest, const T& value) const {
    dest = value;
  }
};

template <typename T>
struct Op_PlusAssign {
  HD_INLINE void operator()(T& dest, const T& value) const {
    dest += value;
  }
};

template <typename T>
struct Op_MinusAssign {
  HD_INLINE void operator()(T& dest, const T& value) const {
    dest -= value;
  }
};

template <typename T>
struct Op_MultAssign {
  HD_INLINE void operator()(T& dest, const T& value) const {
    dest *= value;
  }
};

template <typename T>
struct Op_DivAssign {
  HD_INLINE void operator()(T& dest, const T& value) const {
    if (std::abs(value) < 1.0e-5)
      dest = 0.0f;
    else
      dest /= value;
  }
};

template <typename T>
struct Op_AssignConst {
  T _value;
  HOST_DEVICE Op_AssignConst(const T& value) : _value(value) {}

  HD_INLINE void operator()(T& dest) const { dest = _value; }
};

template <typename T>
struct Op_AssignMultConst
{
  T _value;
  HOST_DEVICE Op_AssignMultConst(const T& value) : _value(value) {}

  HD_INLINE void operator()(T& dest, const T& from) const {
    dest = from * _value;
  }
};

template <typename T>
struct Op_MultConst {
  T _value;
  HOST_DEVICE Op_MultConst(const T& value) : _value(value) {}

  HD_INLINE void operator()(T& dest) const { dest *= _value; }
};

template <typename T>
struct Op_PlusConst {
  T _value;
  HOST_DEVICE Op_PlusConst(const T& value) : _value(value) {}

  HD_INLINE void operator()(T& dest) const { dest += _value; }
};

template <typename T>
struct Op_MinusConst {
  T _value;
  HOST_DEVICE Op_MinusConst(const T& value) : _value(value) {}

  HD_INLINE void operator()(T& dest) const { dest -= _value; }
};

template <typename T>
struct Op_Multiply {
  HD_INLINE T operator()(const T& a, const T& b) const { return a * b; }
};

template <typename T>
struct Op_Plus {
  HD_INLINE T operator()(const T& a, const T& b) const { return a + b; }
};

template <typename T>
struct Op_MultConstRet {
  T _value;
  HOST_DEVICE Op_MultConstRet(const T& value) : _value(value) {}

  HD_INLINE T operator()(const T& a) const { return a * _value; }
};

template <typename T>
struct Op_MultConstAdd {
  T _value;
  HOST_DEVICE Op_MultConstAdd(const T& value) : _value(value) {}

  HD_INLINE void operator()(T& dest, const T& src) const {
    dest += src * _value;
  }
};

template <typename T>
struct Op_AddMultConst {
  T _value;
  HOST_DEVICE Op_AddMultConst(const T& value) : _value(value) {}

  HD_INLINE void operator()(T& dest, const T& src) const {
    dest = dest * _value + src;
  }
};



}

}

#endif  // _OP_H_
