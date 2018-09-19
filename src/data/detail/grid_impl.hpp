#ifndef _GRID_IMPL_HPP_
#define _GRID_IMPL_HPP_

#include <cctype>
#include <iomanip>
#include <iostream>
#include <utility>
// #include <Eigen/Dense>
#include "CudaLE.h"
#include "Integrate.hpp"
#include "algorithms/interpolation.h"
#include "constant_defs.h"
#include "data/grid.h"

// TODO: double check what kind of staggered definition for cell face
// area, determinant, metric elements makes the most sense, and can be
// made backward-compatible

namespace Aperture {

template <typename Metric>
void
Grid::setup_line_cache_f::operator()(const Metric& g,
                                     Grid& grid) const {
  // Loop over the whole grid
  double pos[3];
  auto& mesh = grid.m_mesh;
  for (int k = 0; k < mesh.dims[2]; k++) {
    pos[2] = grid.m_mesh.pos(2, k, 0);
    for (int j = 0; j < mesh.dims[1]; j++) {
      pos[1] = grid.m_mesh.pos(1, j, 0);
      for (int i = 0; i < mesh.dims[0]; i++) {
        pos[0] = mesh.pos(0, i, 0);
        int idx = mesh.get_idx(i, j, k);
        // index 0 is \int \alpha\gamma_{rr} dr
        auto Int = _1D::GQ::GaussLegendreQuadrature<double, 64>();
        grid.m_line_cache[0][idx] = Int(
            [&g, &mesh, &pos](double x) {
              return g.alpha(x, pos[1], pos[2]) *
                     g.g11(x, pos[1], pos[2]);
            },
            pos[0], pos[0] + mesh.delta[0]);
        // index 1 is \int \alpha\gamma_{\theta\theta} d\theta
        grid.m_line_cache[1][idx] = Int(
            [&g, &mesh, &pos](double x) {
              return g.alpha(pos[0], x, pos[2]) *
                     g.g22(pos[0], x, pos[2]);
            },
            pos[1], pos[1] + mesh.delta[1]);
        // index 2 is \int \alpha\gamma_{r\phi} dr past the position of
        // B^r, but it will be multiplied eventually with B^\phi
        grid.m_line_cache[2][idx] = Int(
            [&g, &mesh, &pos](double x) {
              return g.alpha(x, pos[1], pos[2]) *
                     g.g13(x, pos[1], pos[2]);
            },
            pos[0], pos[0] + mesh.delta[0]);
        // index 3 is \int \beta^r\sqrt{\gamma} d\theta, past the
        // position of B^\theta, but this is integrating the edge of our
        // own grid. Draw the grid to see what is going on
        grid.m_line_cache[3][idx] = Int(
            [&g, &mesh, &pos](double x) {
              return g.b1(pos[0] + 0.5 * mesh.delta[0], x, pos[2]) *
                     sqrt(g.det(pos[0] + 0.5 * mesh.delta[0], x,
                                pos[2]));
            },
            pos[1] - 0.5 * mesh.delta[1], pos[1] + 0.5 * mesh.delta[1]);
      }
    }
  }
}

template <typename Metric>
void
Grid::setup_areas_f::operator()(const Metric& g, Grid& grid,
                                int subdivide) const {
  // Loop over the whole grid
  double pos[3];
  auto& mesh = grid.m_mesh;
  for (int k = 0; k < mesh.dims[2]; k++) {
    pos[2] = grid.m_mesh.pos(2, k, 0);
    for (int j = 0; j < mesh.dims[1]; j++) {
      pos[1] = grid.m_mesh.pos(1, j, 0);
      for (int i = 0; i < mesh.dims[0]; i++) {
        pos[0] = mesh.pos(0, i, 0);
        int idx = mesh.get_idx(i, j, k);
        // Then integrate face area
        for (int n = 0; n < 3; n++) {
          double area = 0.0;
          if (n == 0) {
            // r direction
            auto Int = _1D::GQ::GaussLegendreQuadrature<double, 64>();
            if (std::abs(pos[1]) < 1.0e-7) {
              area =
                  2.0 * mesh.delta[2] *
                  Int(
                      [&g, &mesh, &pos](double x) {
                        return sqrt(g.det(pos[0] + 0.5 * mesh.delta[0],
                                          x, pos[2]));
                      },
                      pos[1], pos[1] + 0.5 * mesh.delta[1]);
            } else {
              area =
                  mesh.delta[2] *
                  Int(
                      [&g, &mesh, &pos](double x) {
                        return sqrt(g.det(pos[0] + 0.5 * mesh.delta[0],
                                          x, pos[2]));
                      },
                      pos[1] - 0.5 * mesh.delta[1],
                      pos[1] + 0.5 * mesh.delta[1]);
            }
          } else if (n == 1) {
            // theta direction
            auto Int = _1D::GQ::GaussLegendreQuadrature<double, 64>();
            area = mesh.delta[2] *
                   Int(
                       [&g, &mesh, &pos](double x) {
                         return sqrt(g.det(
                             x, pos[1] + 0.5 * mesh.delta[1], pos[2]));
                       },
                       pos[0] - 0.5 * mesh.delta[0],
                       pos[0] + 0.5 * mesh.delta[0]);
          } else if (n == 2) {
            // phi direction
            auto Int = _2D::GQ::GaussLegendreQuadrature<double, 64>();
            area = Int(
                [&g, &pos](double x, double y) {
                  return sqrt(g.det(x, y, pos[2]));
                },
                pos[0] - 0.5 * mesh.delta[0],
                pos[0] + 0.5 * mesh.delta[0],
                pos[1] - 0.5 * mesh.delta[1],
                pos[1] + 0.5 * mesh.delta[1]);
          }
          grid.m_cell_face_area[n][idx] = area;
        }
      }
    }
  }
}

// template <typename Metric>
// void
// Grid::setup_gamma(const Metric& g) {
//   // Loop over the whole grid
//   double pos[3];
//   for (int k = 0; k < m_mesh.dims[2]; k++) {
//     pos[2] = m_mesh.pos(2, k, 0);
//     for (int j = 0; j < m_mesh.dims[1]; j++) {
//       pos[1] = m_mesh.pos(1, j, 0);
//       for (int i = 0; i < m_mesh.dims[0]; i++) {
//         pos[0] = m_mesh.pos(0, i, 0);
//         int idx = m_mesh.getIdx(i, j, k);
//         // field components are on face centers
//         for (int n = 0; n < 3; n++) {
//           int n_trans[2] = {(n + 1) % 3, (n + 2) % 3};
//           if (n < m_mesh.dim()) {
//             pos[n] += 0.5 * m_mesh.delta[n];
//             for (int m = 0; m < 3; m++) {
//               if (m_metric[n][m].size() > 0) {
//                 m_metric[n][m][idx] = g(n + 1, m + 1, pos);
//               }
//             }
//             pos[n] -= 0.5 * m_mesh.delta[n];
//           } else {
//             for (int m = 0; m < 3; m++) {
//               if (m_metric[n][m].size() > 0)
//                 m_metric[n][m][idx] = g(n + 1, m + 1, pos);
//             }
//           }
//         }
//       }
//     }
//   }
// }

template <typename Metric>
void
Grid::setup_metric_f::operator()(const Metric& g, Grid& grid) const {
  // grid.m_type = (MetricType)Metric::type;
  grid.m_metric_ptr = &g;
  grid.m_type = grid.m_metric_ptr->type();
  // Need to adjust the grid hash according to the metric
  std::size_t metric_hash = grid.hash_line(g.name());
  grid.m_grid_hash <<= 1;
  grid.m_grid_hash ^= metric_hash;
  grid.gen_file_name();

  // Allocate det, alpha, and beta arrays
  grid.allocate_arrays();
  // Mask the corresponding beta array
  if (g.b1 != CudaLE::ZeroOp())
    grid.m_beta1_mask[1] = grid.m_beta2_mask[2] = true;
  if (g.b2 != CudaLE::ZeroOp())
    grid.m_beta1_mask[2] = grid.m_beta2_mask[0] = true;
  if (g.b3 != CudaLE::ZeroOp())
    grid.m_beta1_mask[0] = grid.m_beta2_mask[1] = true;

  // Resize the metric arrays, always do the diagonal
  grid.m_metric[0][0].resize(grid.m_mesh.extent());
  grid.m_metric[1][1].resize(grid.m_mesh.extent());
  grid.m_metric[2][2].resize(grid.m_mesh.extent());
  grid.m_metric_mask[0][0] = grid.m_metric_mask[1][1] =
      grid.m_metric_mask[2][2] = 1;
  // optionally do the non-diagonal depending on spatial metric
  if (g.g12 != CudaLE::ZeroOp()) {
    grid.m_metric[0][1].resize(grid.m_mesh.extent());
    grid.m_metric[1][0].resize(grid.m_mesh.extent());
    grid.m_metric_mask[0][1] = grid.m_metric_mask[1][0] = 1;
  }
  if (g.g13 != CudaLE::ZeroOp()) {
    grid.m_metric[0][2].resize(grid.m_mesh.extent());
    grid.m_metric[2][0].resize(grid.m_mesh.extent());
    grid.m_metric_mask[0][2] = grid.m_metric_mask[2][0] = 1;
  }
  if (g.g23 != CudaLE::ZeroOp()) {
    grid.m_metric[1][2].resize(grid.m_mesh.extent());
    grid.m_metric[2][1].resize(grid.m_mesh.extent());
    grid.m_metric_mask[1][2] = grid.m_metric_mask[2][1] = 1;
  }

  // Loop over the whole grid
  double pos[3];
  for (int k = 0; k < grid.m_mesh.dims[2]; k++) {
    pos[2] = grid.m_mesh.pos(2, k, 0);
    for (int j = 0; j < grid.m_mesh.dims[1]; j++) {
      pos[1] = grid.m_mesh.pos(1, j, 0);
      for (int i = 0; i < grid.m_mesh.dims[0]; i++) {
        pos[0] = grid.m_mesh.pos(0, i, 0);
        int idx = grid.m_mesh.get_idx(i, j, k);
        for (int n = 0; n < 3; n++) {
          int n_trans[2] = {(n + 1) % 3, (n + 2) % 3};
          if (n < grid.m_mesh.dim()) {
            // We define metric coefficients at face centers with
            // respect to the first index
            pos[n] += 0.5 * grid.m_mesh.delta[n];
            for (int m = 0; m < 3; m++) {
              if (grid.m_metric[n][m].size() > 0) {
                grid.m_metric[n][m][idx] = g(n + 1, m + 1, pos);
              }
            }
            // Determinant is also defined at cell faces
            double det = sqrt(g.det(pos));
            // Test for singularity
            // if (std::abs(det) < 1.0e-10) {
            //   pos[1] += 1.0e-5;
            //   det = sqrt(g.det(pos));
            // }
            grid.m_det[n][idx] = det;
            grid.m_alpha[n][idx] = g.alpha(pos);
            grid.m_beta1[n][idx] = g.beta(n_trans[0] + 1, pos);
            grid.m_beta2[n][idx] = g.beta(n_trans[1] + 1, pos);
            pos[n] -= 0.5 * grid.m_mesh.delta[n];
          } else {
            for (int m = 0; m < 3; m++) {
              // For suppressed dimensions (e.g. 3rd dimension in a 2D
              // simulation) we simply don't stagger
              if (grid.m_metric[n][m].size() > 0)
                grid.m_metric[n][m][idx] = g(n + 1, m + 1, pos);
            }
            double det = sqrt(g.det(pos));
            // Test for singularity
            // if (std::abs(det) < 1.0e-10) {
            //   pos[1] += 1.0e-5;
            //   det = sqrt(g.det(pos));
            // }
            grid.m_det[n][idx] = det;
            grid.m_alpha[n][idx] = g.alpha(pos);
            grid.m_beta1[n][idx] = g.beta(n_trans[0] + 1, pos);
            grid.m_beta2[n][idx] = g.beta(n_trans[1] + 1, pos);
          }
        }
      }
    }
  }

  // update grid hash
  grid.m_grid_hash = grid.hash(grid.gen_config());
  grid.gen_file_name();
  // Open file as test
  std::ifstream f_test;
  f_test.open(grid.m_cache_filename.c_str());
  if (f_test.good()) {
    // File exists, read from it
    f_test.close();
    grid.load_from_disk();
  } else {
    // File does not exist, compute new grid cache
    f_test.close();
    grid.setup_areas(g, grid, 64);
    grid.setup_line_cache(g, grid);
    grid.save_to_disk();
  }
  // std::cout << "Min resolved length is " << min_resolved_length() <<
  // std::endl;
}

// template <typename Metric>
// void
// Grid::setup_scales_f::operator()(const Metric& g, Grid& grid) const {
//   auto mesh = grid.m_mesh;
//   int dim = mesh.dim();
//   int stagger_num = (1 << dim);
//   for (int i = 0; i < 3; i++) {
//     grid.m_scales[i].resize(stagger_num);
//     for (int j = 0; j < stagger_num; j++) {
//       grid.m_scales[i][j].resize(mesh.extent());
//     }
//   }

//   for (int n = 0; n < stagger_num; n++) {
//     // Using bit operations to be more intuitive
//     int istag = check_bit(n, 0); // Extract lowest bit
//     int jstag = check_bit(n, 1); // Extract second lowest bit
//     int kstag = check_bit(n, 2); // Extract highest bit
//     // loop over cell indices(i,j,k)
//     for (int k = 0; k < mesh.dims[2]; k++) {
//       for (int j = 0; j < mesh.dims[1]; j++) {
//         for (int i = 0; i < mesh.dims[0]; i++) {
//           // calculate the coordinate values
//           double q1 = mesh.pos(0, i, istag) + EPS;
//           double q2 = mesh.pos(1, j, jstag) + EPS;
//           double q3 = mesh.pos(2, k, kstag) + EPS;
//           // calculate the scaling functions h1, h2, h3
//           grid.m_scales[0][n](i, j, k) = std::sqrt(g.g11(q1, q2,
//           q3)); grid.m_scales[1][n](i, j, k) = std::sqrt(g.g22(q1,
//           q2, q3)); grid.m_scales[2][n](i, j, k) =
//           std::sqrt(g.g33(q1, q2, q3));
//         }
//       }
//     }

//   }
// }

// template <typename Metric>
// void
// Grid::setup_connection(const Metric& g) {
//   // Loop over the whole grid
//   double pos[3];
//   Eigen::Matrix4d g_uv, g_inv;
//   for (int k = 0; k < m_mesh.dims[2]; k++) {
//     pos[2] = m_mesh.pos(2, k, 0);
//     for (int j = 0; j < m_mesh.dims[1]; j++) {
//       pos[1] = m_mesh.pos(1, j, 0);
//       for (int i = 0; i < m_mesh.dims[0]; i++) {
//         pos[0] = m_mesh.pos(0, i, 0);
//         // connection components are at cell centers
//         // we first construct the matrix g_{\mu\nu}
//         for (int n = 0; n < 4; n++) {
//           for (int m = 0; m < 4; m++) {
//             g_uv(n, m) = g(n, m, pos);
//             // g_0m += g_uv(n, m) * g.beta(m, pos);
//           }
//         }
//         g_inv = g_uv.inverse();
//         if (i == 20 && j == 20 && k == 0) {
//           std::cout << g_inv << std::endl;
//         }

//         int idx = m_mesh.getIdx(i, j, k);
//         for (int mu = 0; mu < 4; mu++) {
//           // This is the only loop we can iterate over
//           // First index is i, second is alpha, in the expression
//           // g_{\alpha\beta, i}g^{\beta\mu}
//           // m_connection[0][0][mu] = PARTIAL_G(g,0,0,1,pos) *
//           g_inv(0, mu); m_connection[0][0][mu][idx] =
//           SUM_PARTIAL_G(g,0,0,mu,g_inv,pos); if
//           (m_connection[0][0][mu][idx] != 0.0)
//           m_connection_mask[0][0][mu] = 1;
//           m_connection[0][1][mu][idx] =
//           SUM_PARTIAL_G(g,1,0,mu,g_inv,pos); if
//           (m_connection[0][1][mu][idx] != 0.0)
//           m_connection_mask[0][1][mu] = 1;
//           m_connection[0][2][mu][idx] =
//           SUM_PARTIAL_G(g,2,0,mu,g_inv,pos); if
//           (m_connection[0][2][mu][idx] != 0.0)
//           m_connection_mask[0][2][mu] = 1;
//           m_connection[0][3][mu][idx] =
//           SUM_PARTIAL_G(g,3,0,mu,g_inv,pos); if
//           (m_connection[0][3][mu][idx] != 0.0)
//           m_connection_mask[0][3][mu] = 1;
//           m_connection[1][0][mu][idx] =
//           SUM_PARTIAL_G(g,0,1,mu,g_inv,pos); if
//           (m_connection[1][0][mu][idx] != 0.0)
//           m_connection_mask[1][0][mu] = 1;
//           m_connection[1][1][mu][idx] =
//           SUM_PARTIAL_G(g,1,1,mu,g_inv,pos); if
//           (m_connection[1][1][mu][idx] != 0.0)
//           m_connection_mask[1][1][mu] = 1;
//           m_connection[1][2][mu][idx] =
//           SUM_PARTIAL_G(g,2,1,mu,g_inv,pos); if
//           (m_connection[1][2][mu][idx] != 0.0)
//           m_connection_mask[1][2][mu] = 1;
//           m_connection[1][3][mu][idx] =
//           SUM_PARTIAL_G(g,3,1,mu,g_inv,pos); if
//           (m_connection[1][3][mu][idx] != 0.0)
//           m_connection_mask[1][3][mu] = 1;
//           // By default the simulation is 2D, therefore third
//           derivative is
//           // always zero because we assume it's symmetry direction
//           if (m_mesh.dim() > 2) {
//             m_connection[2][0][mu][idx] =
//             SUM_PARTIAL_G(g,0,2,mu,g_inv,pos); if
//             (m_connection[2][0][mu][idx] != 0.0)
//             m_connection_mask[2][0][mu] = 1;
//             m_connection[2][1][mu][idx] =
//             SUM_PARTIAL_G(g,1,2,mu,g_inv,pos); if
//             (m_connection[2][1][mu][idx] != 0.0)
//             m_connection_mask[2][1][mu] = 1;
//             m_connection[2][2][mu][idx] =
//             SUM_PARTIAL_G(g,2,2,mu,g_inv,pos); if
//             (m_connection[2][2][mu][idx] != 0.0)
//             m_connection_mask[2][2][mu] = 1;
//             m_connection[2][3][mu][idx] =
//             SUM_PARTIAL_G(g,3,2,mu,g_inv,pos); if
//             (m_connection[2][3][mu][idx] != 0.0)
//             m_connection_mask[2][3][mu] = 1;
//           }
//         }
//       }
//     }
//   }
// }

template <typename Metric>
Grid
Grid::make_dual_f::operator()(const Metric& g, bool inverse) const {
  Grid grid = grid.make_dual(inverse);
  grid.setup_metric(g, grid);
  return grid;
}

// template <int InterpolationOrder, typename POS_T>
// Matrix3
// Grid::metric_matrix(int cell, const Vec3<POS_T>& rel_pos) const {
//   interpolator<InterpolationOrder> interp;
//   Matrix3 result = Matrix3::ZeroOp();

//   Vec3<int> c = m_mesh.getCell3D(cell);
//   Vec3<int> lower = c - interp.radius();
//   Vec3<int> upper = c + interp.support() - interp.radius();
//   if (dim() < 3) {
//     lower[2] = upper[2] = c[2];
//   }
//   for (int n = 0; n < 3; n++) {
//     for (int m = 0; m < 3; m++) {
//       if (m_metric_mask[n][m] != 1) continue;
//       for (int k = lower[2]; k <= upper[2]; k++) {
//         for (int j = lower[1]; j <= upper[1]; j++) {
//           for (int i = lower[0]; i <= upper[0]; i++) {
//             if (dim() < 3) {
//               // if (m_metric[n][m].size() > 0) {
//                 result(n, m) += m_metric[n][m](i, j, k) *
//                 interp.interp_cell(rel_pos[0], c[0], i, (n == 0 ? 1 :
//                 0))
//                                 * interp.interp_cell(rel_pos[1],
//                                 c[1], j, (n == 1 ? 1 : 0));
//               // }
//             } else {
//               result(n, m) += m_metric[n][m](i, j, k) *
//               interp.interp_cell(rel_pos[0], c[0], i, (n == 0 ? 1 :
//               0))
//                               * interp.interp_cell(rel_pos[1], c[1],
//                               j, (n == 1 ? 1 : 0))
//                               * interp.interp_cell(rel_pos[2], c[2],
//                               k, (n == 2 ? 1 : 0));
//             }
//           }
//         }
//       }
//     }
//   }
//   return result;
// }

// template <int InterpolationOrder, typename POS_T>
// Matrix3
// Grid::metric_matrix(const Vec3<POS_T>& pos) const {
//   Vec3<POS_T> rel_pos;
//   int cell = m_mesh.findCell(pos, rel_pos);
//   return metric_matrix<InterpolationOrder>(cell, rel_pos);
// }

// template <int InterpolationOrder, typename POS_T>
// Scalar
// Grid::alpha(int cell, const Vec3<POS_T>& rel_pos) const {
//   interpolator<InterpolationOrder> interp;
//   // Original metric array has stagger same as vector component n
//   Scalar result = 0.0;

//   Vec3<int> c = m_mesh.getCell3D(cell);
//   Vec3<int> lower = c - interp.radius();
//   Vec3<int> upper = c + interp.support() - interp.radius();
//   if (dim() < 3) {
//     lower[2] = upper[2] = c[2];
//   }
//   for (int k = lower[2]; k <= upper[2]; k++) {
//     for (int j = lower[1]; j <= upper[1]; j++) {
//       for (int i = lower[0]; i <= upper[0]; i++) {
//         result += m_alpha[0](i, j, k) *
//         interp.interp_cell(rel_pos[0], c[0], i, 1)
//                   * interp.interp_cell(rel_pos[1], c[1], j, 0)
//                   * (dim() < 3 ? 1.0 : interp.interp_cell(rel_pos[2],
//                   c[2], k, 0));
//       }
//     }
//   }
//   return result;
// }

// template <int InterpolationOrder, typename POS_T>
// Scalar
// Grid::alpha(const Vec3<POS_T>& pos) const {
//   Vec3<POS_T> rel_pos;
//   int cell = m_mesh.findCell(pos, rel_pos);
//   return alpha<InterpolationOrder>(cell, rel_pos);
// }

// template <int InterpolationOrder, typename POS_T>
// Scalar
// Grid::det(int cell, const Vec3<POS_T>& rel_pos) const {
//   interpolator<InterpolationOrder> interp;
//   // Original metric array has stagger same as vector component n
//   Scalar result = 0.0;

//   Vec3<int> c = m_mesh.getCell3D(cell);
//   Vec3<int> lower = c - interp.radius();
//   Vec3<int> upper = c + interp.support() - interp.radius();
//   if (dim() < 3) {
//     lower[2] = upper[2] = c[2];
//   }
//   for (int k = lower[2]; k <= upper[2]; k++) {
//     for (int j = lower[1]; j <= upper[1]; j++) {
//       for (int i = lower[0]; i <= upper[0]; i++) {
//         result += m_det[0](i, j, k) * interp.interp_cell(rel_pos[0],
//         c[0], i, 1)
//                   * interp.interp_cell(rel_pos[1], c[1], j, 0)
//                   * (dim() < 3 ? 1.0 : interp.interp_cell(rel_pos[2],
//                   c[2], k, 0));
//       }
//     }
//   }
//   return result;
// }

// template <int InterpolationOrder, typename POS_T>
// Scalar
// Grid::det(const Vec3<POS_T>& pos) const {
//   Vec3<POS_T> rel_pos;
//   int cell = m_mesh.findCell(pos, rel_pos);
//   return det<InterpolationOrder>(cell, rel_pos);
// }

// template <int InterpolationOrder, typename POS_T>
// Vector3
// Grid::beta(int cell, const Vec3<POS_T>& rel_pos) const {
//   interpolator<InterpolationOrder> interp;
//   // Original metric array has stagger same as vector component n
//   Vector3 result(0.0, 0.0, 0.0);

//   Vec3<int> c = m_mesh.getCell3D(cell);
//   Vec3<int> lower = c - interp.radius();
//   Vec3<int> upper = c + interp.support() - interp.radius();
//   if (dim() < 3) {
//     lower[2] = upper[2] = c[2];
//   }
//   for (int k = lower[2]; k <= upper[2]; k++) {
//     for (int j = lower[1]; j <= upper[1]; j++) {
//       for (int i = lower[0]; i <= upper[0]; i++) {
//         if (dim() < 3) {
//           result[0] += m_beta2[1](i, j, k) *
//           interp.interp_cell(rel_pos[0], c[0], i, 0)
//                        * interp.interp_cell(rel_pos[1], c[1], j, 1);
//           result[1] += m_beta1[0](i, j, k) *
//           interp.interp_cell(rel_pos[0], c[0], i, 1)
//                        * interp.interp_cell(rel_pos[1], c[1], j, 0);
//           result[2] += m_beta2[0](i, j, k) *
//           interp.interp_cell(rel_pos[0], c[0], i, 1)
//                        * interp.interp_cell(rel_pos[1], c[1], j, 0);
//         } else {
//           result[0] += m_beta1[2](i, j, k) *
//           interp.interp_cell(rel_pos[0], c[0], i, 0)
//                        * interp.interp_cell(rel_pos[1], c[1], j, 0)
//                        * interp.interp_cell(rel_pos[2], c[2], k, 1);
//           result[1] += m_beta1[0](i, j, k) *
//           interp.interp_cell(rel_pos[0], c[0], i, 1)
//                        * interp.interp_cell(rel_pos[1], c[1], j, 0)
//                        * interp.interp_cell(rel_pos[2], c[2], k, 0);
//           result[2] += m_beta1[1](i, j, k) *
//           interp.interp_cell(rel_pos[0], c[0], i, 0)
//                        * interp.interp_cell(rel_pos[1], c[1], j, 1)
//                        * interp.interp_cell(rel_pos[2], c[2], k, 0);
//         }
//       }
//     }
//   }
//   return result;
// }

// template <int InterpolationOrder, typename POS_T>
// Vector3
// Grid::beta(const Vec3<POS_T>& pos) const {
//   Vec3<POS_T> rel_pos;
//   int cell = m_mesh.findCell(pos, rel_pos);
//   return beta<InterpolationOrder>(cell, rel_pos);
// }

// template <int InterpolationOrder, typename POS_T>
// void
// Grid::connection(int cell, const Vec3<POS_T>& rel_pos, double
// conn[3][4][4]) const {
//   interpolator<InterpolationOrder> interp;
//   // All connections are evaluated at the center of the grid

//   Vec3<int> c = m_mesh.getCell3D(cell);
//   Vec3<int> lower = c - interp.radius();
//   Vec3<int> upper = c + interp.support() - interp.radius();
//   if (dim() < 3) {
//     lower[2] = upper[2] = c[2];
//   }
//   for (int a = 0; a < 3; a++) {
//     for (int m = 0; m < 4; m++) {
//       for (int n = 0; n < 4; n++) {
//         conn[a][m][n] = 0.0;
//       }
//     }
//   }
//   for (int a = 0; a < 3; a++) {
//     for (int m = 0; m < 4; m++) {
//       for (int n = 0; n < 4; n++) {
//         if (m_connection_mask[a][m][n] != 1) continue;
//         for (int k = lower[2]; k <= upper[2]; k++) {
//           for (int j = lower[1]; j <= upper[1]; j++) {
//             for (int i = lower[0]; i <= upper[0]; i++) {
//               conn[a][m][n] += m_connection[a][m][n](i, j, k) *
//               interp.interp_cell(rel_pos[0], c[0], i, 0)
//                                * interp.interp_cell(rel_pos[1], c[1],
//                                j, 0)
//                                * (dim() < 3 ? 1.0 :
//                                interp.interp_cell(rel_pos[2], c[2],
//                                k, 0));
//             }
//           }
//         }
//       }
//     }
//   }
// }

// template <int InterpolationOrder, typename POS_T>
// void
// Grid::connection(const Vec3<POS_T>& pos, double conn[3][4][4]) const
// {
//   Vec3<POS_T> rel_pos;
//   int cell = m_mesh.findCell(pos, rel_pos);
//   connection<InterpolationOrder>(cell, rel_pos, conn);
// }

// Scalar
// Grid::min_resolved_length(int cell) const {
//   // Scalar lambda_p = std::sqrt(m_metric[0][0][cell]) *
//   m_mesh.delta[0];
//   // if (dim() > 1) {
//   //   Scalar tmp =  std::sqrt(m_metric[1][1][cell]) *
//   m_mesh.delta[1];
//   //   lambda_p = lambda_p > tmp ? lambda_p : tmp;
//   // }
//   // if (dim() > 2) {
//   //   Scalar tmp =  std::sqrt(m_metric[2][2][cell]) *
//   m_mesh.delta[2];
//   //   lambda_p = lambda_p > tmp ? lambda_p : tmp;
//   // }
//   double lambda_p = std::min(m_mesh.delta[0], m_mesh.delta[1]);
//   return lambda_p;
// }

// Scalar
// Grid::cell_volume(int cell) const {
//   return m_det[2][cell] * m_mesh.delta[0] * m_mesh.delta[1] * (dim()
//   > 2 ? m_mesh.delta[2] : 1.0);
// }

}  // namespace Aperture

#ifdef SUM_PARTIAL_G
#undef SUM_PARTIAL_G
#endif

#endif  // _GRID_IMPL_HPP_
