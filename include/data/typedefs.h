#ifndef  _TYPEDEFS_H_
#define  _TYPEDEFS_H_

#include <bitset>
#include <cstddef>
// #include <Eigen/Dense>

namespace Aperture {

typedef double Scalar;

/// A simple way to track staggering of field components. Note that when
/// initializing it, one can use Stagger_t("011"). The rightmost digit is for x
/// direction, while leftmost digit for z
typedef std::bitset<3> Stagger_t;

typedef double Mom_t;
typedef double Pos_t;

typedef std::size_t Index_t;

// typedef Eigen::Matrix<Scalar, 3, 3> Matrix3;
// typedef Eigen::Matrix<Scalar, 4, 4> Matrix4;
// typedef Eigen::Matrix<Scalar, 3, 1> Vector3;
// typedef Eigen::Matrix<Scalar, 4, 1> Vector4;

}

#endif   // _TYPEDEFS_H_
