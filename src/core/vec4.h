#ifndef _VEC4_H_
#define _VEC4_H_

#include "core/vec3.h"

namespace Aperture {

/// Vectorized type as 4d vector, better alignment properties than 3d
/// vector
template <typename T>
struct Vec4 {
  T x, y, z, w;

  typedef Vec4<T> self_type;

  HOST_DEVICE Vec4()
      : x(static_cast<T>(0)),
        y(static_cast<T>(0)),
        z(static_cast<T>(0)),
        w(static_cast<T>(0)) {}
  HOST_DEVICE Vec4(T xi, T yi, T zi, T wi = static_cast<T>(0))
      : x(xi), y(yi), z(zi), w(wi) {}
  HOST_DEVICE Vec4(const Vec3<T>& other) {
    x = other.x;
    y = other.y;
    z = other.z;
    w = static_cast<T>(0);
  }
  template <typename U>
  HOST_DEVICE Vec4(const Vec4<U>& other) {
    x = static_cast<T>(other.x);
    y = static_cast<T>(other.y);
    z = static_cast<T>(other.z);
    w = static_cast<T>(other.w);
  }

  HOST_DEVICE Vec4(const self_type& other) = default;
  HOST_DEVICE Vec4(self_type&& other) = default;

  HD_INLINE T& operator[](int idx) {
    if (idx == 0)
      return x;
    else if (idx == 1)
      return y;
    else if (idx == 2)
      return z;
    else
      return w;
    // else
    //   throw std::out_of_range("Index out of bound!");
  }

  HD_INLINE const T& operator[](int idx) const {
    if (idx == 0)
      return x;
    else if (idx == 1)
      return y;
    else if (idx == 2)
      return z;
    else
      return w;
    // else
    //   throw std::out_of_range("Index out of bound!");
  }

  HD_INLINE self_type& operator=(const self_type& other) {
    x = other.x;
    y = other.y;
    z = other.z;
    w = other.w;
    return *this;
  }

  HD_INLINE bool operator==(const self_type& other) const {
    return (x == other.x && y == other.y && z == other.z &&
            w == other.w);
  }

  HD_INLINE bool operator!=(const self_type& other) const {
    return (x != other.x || y != other.y || z != other.z ||
            w != other.w);
  }

  HD_INLINE self_type& operator+=(const self_type& other) {
    x += other.x;
    y += other.y;
    z += other.z;
    w += other.w;
    return (*this);
  }

  HD_INLINE self_type& operator-=(const self_type& other) {
    x -= other.x;
    y -= other.y;
    z -= other.z;
    w -= other.w;
    return (*this);
  }

  HD_INLINE self_type& operator+=(T n) {
    x += n;
    y += n;
    z += n;
    w += n;
    return (*this);
  }

  HD_INLINE self_type& operator-=(T n) {
    x -= n;
    y -= n;
    z -= n;
    w -= n;
    return (*this);
  }

  HD_INLINE self_type& operator*=(T n) {
    x *= n;
    y *= n;
    z *= n;
    w *= n;
    return (*this);
  }

  HD_INLINE self_type& operator/=(T n) {
    x /= n;
    y /= n;
    z /= n;
    w /= n;
    return (*this);
  }

  HD_INLINE self_type operator+(const self_type& other) const {
    self_type tmp = *this;
    tmp += other;
    return tmp;
  }

  HD_INLINE self_type operator-(const self_type& other) const {
    self_type tmp = *this;
    tmp -= other;
    return tmp;
  }

  HD_INLINE self_type operator+(T n) const {
    self_type tmp = *this;
    tmp += n;
    return tmp;
  }

  HD_INLINE self_type operator-(T n) const {
    self_type tmp = *this;
    tmp -= n;
    return tmp;
  }

  HD_INLINE self_type operator*(T n) const {
    // self_type tmp{x * n, y * n, z * n};
    return {x * n, y * n, z * n, w * n};
  }

  HD_INLINE self_type operator/(T n) const {
    // self_type tmp{x / n, y / n, z / n};
    return {x / n, y / n, z / n, w / n};
  }

  HD_INLINE Vec3<T> vec3() const { return {x, y, z}; }

  friend std::ostream& operator<<(std::ostream& os,
                                  const self_type& vec) {
    os << "( " << vec.x << ", " << vec.y << ", " << vec.z << ", "
       << vec.w << " )";
    return os;
  }
};

template <typename T>
Vec4<T> operator*(const T& t, const Vec4<T>& v) {
  Vec4<T> result(v);
  result *= t;
  return result;
}

// template <typename T>
// Vector4 to_eigen(const Vec4<T>& vec) {
//   Vector4 result(vec.x, vec.y, vec.z, vec.w);
//   return result;
// }

// template <typename T>
// Vec4<T> from_eigen(const Vector4& vec) {
//   Vec4<T> result(vec[0], vec[1], vec[2], vec[3]);
//   return result;
// }

}  // namespace Aperture

#endif  // _VEC4_H_
