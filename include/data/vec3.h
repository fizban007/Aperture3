#ifndef _VEC3_H_
#define _VEC3_H_

#include <cmath>
#include <iostream>
// #include <mpi.h>
#include "data/typedefs.h"
// #include "cuda/cuda_control.h"

namespace Aperture {

///  Vectorized type as 3d vector
template <typename T>
struct Vec3 {
  T x, y, z;

  typedef Vec3<T> self_type;

  Vec3()
      : x(static_cast<T>(0)), y(static_cast<T>(0)), z(static_cast<T>(0)) {}
  Vec3(T xi, T yi, T zi) : x(xi), y(yi), z(zi) {}
  template <typename U>
  Vec3(const Vec3<U>& other) {
    x = static_cast<T>(other.x);
    y = static_cast<T>(other.y);
    z = static_cast<T>(other.z);
  }

  Vec3(const self_type& other) = default;
  Vec3(self_type&& other) = default;

  T& operator[](int idx) {
    if (idx == 0)
      return x;
    else if (idx == 1)
      return y;
    // else if (idx == 2)
    else
      return z;
    // else
    //   throw std::out_of_range("Index out of bound!");
  }

  const T& operator[](int idx) const {
    if (idx == 0)
      return x;
    else if (idx == 1)
      return y;
    // else if (idx == 2)
    else
      return z;
    // else
    // throw std::out_of_range("Index out of bound!");
  }

  self_type& operator=(const self_type& other) {
    x = other.x;
    y = other.y;
    z = other.z;
    return *this;
  }

  bool operator==(const self_type& other) const {
    return (x == other.x && y == other.y && z == other.z);
  }

  bool operator!=(const self_type& other) const {
    return (x != other.x || y != other.y || z != other.z);
  }

  self_type& operator+=(const self_type& other) {
    x += other.x;
    y += other.y;
    z += other.z;
    return (*this);
  }

  self_type& operator-=(const self_type& other) {
    x -= other.x;
    y -= other.y;
    z -= other.z;
    return (*this);
  }

  self_type& operator+=(T n) {
    x += n;
    y += n;
    z += n;
    return (*this);
  }

  self_type& operator-=(T n) {
    x -= n;
    y -= n;
    z -= n;
    return (*this);
  }

  self_type& operator*=(T n) {
    x *= n;
    y *= n;
    z *= n;
    return (*this);
  }

  self_type& operator/=(T n) {
    x /= n;
    y /= n;
    z /= n;
    return (*this);
  }

  self_type operator+(const self_type& other) const {
    self_type tmp = *this;
    tmp += other;
    return tmp;
  }

  self_type operator-(const self_type& other) const {
    self_type tmp = *this;
    tmp -= other;
    return tmp;
  }

  self_type operator+(T n) const {
    self_type tmp = *this;
    tmp += n;
    return tmp;
  }

  self_type operator-(T n) const {
    self_type tmp = *this;
    tmp -= n;
    return tmp;
  }

  self_type operator*(T n) const {
    self_type tmp{x * n, y * n, z * n};
    return tmp;
  }

  self_type operator/(T n) const {
    self_type tmp{x / n, y / n, z / n};
    return tmp;
  }

  T dot(const self_type& other) const {
    return (x * other.x + y * other.y + z * other.z);
  }

  self_type cross(const self_type& other) const {
    return Vec3<T>(y * other.z - z * other.y, z * other.x - x * other.z,
                   x * other.y - y * other.x);
  }

  T length() const {
    return std::sqrt(this -> dot(*this));
  }

  self_type& normalize() {
    double l = this -> length();
    if (l > 1.0e-13)
      *this /= l;
    return *this;
  }

  self_type& normalize(T l) {
    this -> normalize();
    (*this) *= l;
    return *this;
  }

  friend std::ostream& operator<<(std::ostream& os, const self_type& vec) {
    os << "( " << vec.x << ", " << vec.y << ", " << vec.z << " )";
    return os;
  }
};

////////////////////////////////////////////////////////////////////////////////
///  Class to store a multi-dimensional size
////////////////////////////////////////////////////////////////////////////////
struct Extent : public Vec3<int> {
  // Default initialize to 0 in the first size and 1 in the rest for
  // safety reasons
  Extent() : Vec3(0, 1, 1) {}
  Extent(int w, int h = 1, int d = 1) : Vec3(w, h, d) {}
  Extent(const Vec3<int>& vec) : Vec3(vec) {}

  int& width() { return x; }
  const int& width() const { return x; }
  int& height() { return y; }
  const int& height() const { return y; }
  int& depth() { return z; }
  const int& depth() const { return z; }

  int size() const { return x * y * z; }
};

////////////////////////////////////////////////////////////////////////////////
///  Class to store a multi-dimensional index
////////////////////////////////////////////////////////////////////////////////
struct Index : public Vec3<int> {
  // Default initialize to 0, first index
  Index() : Vec3(0, 0, 0) {}
  Index(int idx, const Extent& ext) {
    z = idx / (ext.width() * ext.height());
    int slice = idx % (ext.width() * ext.height());
    y = slice / ext.width();
    x = slice % ext.width();
  }
  Index(int xi, int yi, int zi) : Vec3(xi, yi, zi) {}
  Index(const Vec3<int>& vec) : Vec3(vec) {}

  int index(const Extent& ext) const {
    return x + y * ext.width() + z * ext.width() * ext.height();
  }
};

template <typename T>
Vec3<T> operator* (const T& t, const Vec3<T>& v) {
  Vec3<T> result(v);
  result *= t;
  return result;
}

template <typename T>
T abs(const Vec3<T>& v) {
  // return v.length();
  return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

// template <typename T>
// Vector3 to_eigen(const Vec3<T>& vec) {
//   Vector3 result(vec.x, vec.y, vec.z);
//   return result;
// }

// template <typename T>
// Vec3<T> from_eigen(const Vector3& vec) {
//   Vec3<T> result(vec[0], vec[1], vec[2]);
//   return result;
// }

}

#endif  // _VEC3_H_
