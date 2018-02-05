#ifndef _SOLVE_H_
#define _SOLVE_H_

#include "data/vec3.h"
#include <array>

namespace Aperture {

void solve(const std::array<std::array<double, 6>, 6>& J,
           const std::array<double, 6>& F, std::array<double, 6>& result);

void solve(const std::array<std::array<double, 3>, 3>& J,
           const Vec3<double>& F, Vec3<double>& result);

double norm(const std::array<double, 6>& f);

// template <typename Double>
// std::array<std::array<Double, 3>, 3> invert(
//     const std::array<std::array<Double, 3>, 3>& metric);

template <typename Double>
std::array<std::array<Double, 3>, 3> invert(
    const std::array<std::array<Double, 3>, 3>& metric) {
  std::array<std::array<Double, 3>, 3> result;

  Double det = metric[0][0] * (metric[1][1] * metric[2][2] - metric[1][2] * metric[2][1])
               + metric[0][1] * (metric[1][2] * metric[2][0] - metric[1][0] * metric[2][2])
               + metric[0][2] * (metric[1][0] * metric[2][1] - metric[1][1] * metric[2][0]);

  for (int i = 0; i < 3; i++) {
    int minor_i[2] = {(i + 1) % 3, (i + 2) % 3};
    for (int j = 0; j < 3; j++) {
      int minor_j[2] = {(j + 1) % 3, (j + 2) % 3};
      result[i][j] = metric[minor_i[0]][minor_j[0]] * metric[minor_i[1]][minor_j[1]]
                     - metric[minor_i[0]][minor_j[1]] * metric[minor_i[1]][minor_j[0]];
      result[i][j] /= det;
    }
  }

  return result;
}

template <typename Double>
std::array<std::array<Double, 3>, 3> invert(
    const std::array<std::array<Double, 3>, 3>& metric, const Double& det) {
  std::array<std::array<Double, 3>, 3> result;

  for (int i = 0; i < 3; i++) {
    int minor_i[2] = {(i + 1) % 3, (i + 2) % 3};
    for (int j = 0; j < 3; j++) {
      int minor_j[2] = {(j + 1) % 3, (j + 2) % 3};
      result[i][j] = metric[minor_i[0]][minor_j[0]] * metric[minor_i[1]][minor_j[1]]
                     - metric[minor_i[0]][minor_j[1]] * metric[minor_i[1]][minor_j[0]];
      result[i][j] /= det;
    }
  }

  return result;
}

}

#endif  // _SOLVE_H_
