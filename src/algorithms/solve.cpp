#include "algorithms/solve.h"
// #include <Eigen/Dense>
#include <iostream>

namespace Aperture {

// void solve(const std::array<std::array<double, 6>, 6>& J,
//            const std::array<double, 6>& F, std::array<double, 6>&
//            result) {
//   Eigen::MatrixXd J_mat(6, 6);
//   Eigen::VectorXd F_vec(6);
//   for (int m = 0; m < 6; m++) {
//     for (int n = 0; n < 6; n++) {
//       J_mat(n, m) = J[n][m];
//     }
//     // F(m) = f[m].x();
//     F_vec(m) = F[m];
//   }
//   // std::cout << J_mat << std::endl << std::endl;
//   Eigen::VectorXd new_f = J_mat.colPivHouseholderQr().solve(F_vec);

//   for (int i = 0; i < 6; i++) {
//     result[i] = new_f(i);
//   }
// }

void
invert(const std::array<std::array<double, 3>, 3>& A,
       std::array<std::array<double, 3>, 3>& iA) {
  double det;

  det = A[0][0] * (A[2][2] * A[1][1] - A[2][1] * A[1][2]) -
        A[1][0] * (A[2][2] * A[0][1] - A[2][1] * A[0][2]) +
        A[2][0] * (A[1][2] * A[0][1] - A[1][1] * A[0][2]);
  // if( fabs(det) < SOLVE_EPSILON )
  //  return SOLVE_FAILURE;

  iA[0][0] = (A[2][2] * A[1][1] - A[2][1] * A[1][2]) / det;
  iA[0][1] = -(A[2][2] * A[0][1] - A[2][1] * A[0][2]) / det;
  iA[0][2] = (A[1][2] * A[0][1] - A[1][1] * A[0][2]) / det;

  iA[1][0] = -(A[2][2] * A[1][0] - A[2][0] * A[1][2]) / det;
  iA[1][1] = (A[2][2] * A[0][0] - A[2][0] * A[0][2]) / det;
  iA[1][2] = -(A[1][2] * A[0][0] - A[1][0] * A[0][2]) / det;

  iA[2][0] = (A[2][1] * A[1][0] - A[2][0] * A[1][1]) / det;
  iA[2][1] = -(A[2][1] * A[0][0] - A[2][0] * A[0][1]) / det;
  iA[2][2] = (A[1][1] * A[0][0] - A[1][0] * A[0][1]) / det;
}

void
solve(const std::array<std::array<double, 3>, 3>& J,
      const Vec3<double>& b, Vec3<double>& x) {
  std::array<std::array<double, 3>, 3> iA;
  invert(J, iA);

  x[0] = iA[0][0] * b[0] + iA[0][1] * b[1] + iA[0][2] * b[2];
  x[1] = iA[1][0] * b[0] + iA[1][1] * b[1] + iA[1][2] * b[2];
  x[2] = iA[2][0] * b[0] + iA[2][1] * b[1] + iA[2][2] * b[2];
}

// double norm(const std::array<double, 6>& f) {
//   Eigen::VectorXd v(6);
//   for (int i = 0; i < 6; i++) {
//     v[i] = f[i];
//   }
//   return v.norm();
// }

}  // namespace Aperture
