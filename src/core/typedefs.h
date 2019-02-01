#ifndef _TYPEDEFS_H_
#define _TYPEDEFS_H_

#include <bitset>
#include <cstddef>
// #include <Eigen/Dense>

namespace Aperture {

#ifndef USE_DOUBLE
typedef float Scalar;
typedef float Mom_t;
typedef float Pos_t;
#else
typedef double Scalar;
typedef double Mom_t;
typedef double Pos_t;
#endif

typedef std::size_t Index_t;

// typedef Eigen::Matrix<Scalar, 3, 3> Matrix3;
// typedef Eigen::Matrix<Scalar, 4, 4> Matrix4;
// typedef Eigen::Matrix<Scalar, 3, 1> Vector3;
// typedef Eigen::Matrix<Scalar, 4, 1> Vector4;

}  // namespace Aperture

#endif  // _TYPEDEFS_H_