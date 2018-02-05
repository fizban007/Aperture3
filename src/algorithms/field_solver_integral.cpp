#include "field_solver.h"

using namespace Aperture;

FieldSolver_Integral::FieldSolver_Integral(const Grid &g, const Grid &g_dual)
    : m_dE(g), m_dB(g_dual), m_ks_rhs(g.extent()), m_E_aux(g), m_H_aux(g) {
  m_tri_a.resize(g.mesh().dims[0], 0.0);
  m_tri_c.resize(g.mesh().dims[0], 0.0);
  m_tri_d.resize(g.mesh().dims[0], 0.0);
}

FieldSolver_Integral::~FieldSolver_Integral() {}

void
FieldSolver_Integral::update_fields(vfield_t &E, vfield_t &B, const vfield_t &J,
                                    double dt, double time) {
  // Logger::print_info("Updating fields");
  auto &grid = E.grid();
  auto &mesh = grid.mesh();
  // Explicit update
  if (grid.dim() == 1) {
    for (int i = 0; i < mesh.dims[0]; i++) {
      // TODO: Add a background J?
      E(0, i) += -dt * J(0, i);
    }
  }
}

void
FieldSolver_Integral::update_fields(Aperture::SimData &data, double dt,
                                    double time) {
  update_fields(data.E, data.B, data.J, dt, time);
}

// void
// FieldSolver_Integral::compute_B_update(vfield_t &delta_B, const vfield_t &E,
//                                        double dt) {
//   // loop over the grid, skipping the outer most index
//   double dB0, dB1, dB2;
//   delta_B.initialize();
//   // Need to consider how to get in alpha and beta functions
//   auto &grid = E.grid();
//   if (grid.dim() == 2) {
//     // 2D, only do 2 loops
//     for (int j = 1; j < grid.extent().height() - 1; j++) {
//       for (int i = 1; i < grid.extent().width() - 1; i++) {
//         dB2 = E(1, i + 1, j) * delta_B.grid().mesh().delta[1] -
//               E(1, i, j) * delta_B.grid().mesh().delta[1] +
//               E(0, i, j) * delta_B.grid().mesh().delta[0] -
//               E(0, i, j + 1) * delta_B.grid().mesh().delta[0];
//         dB1 = E(2, i, j + 1) * delta_B.grid().mesh().delta[2] -
//               E(2, i + 1, j + 1) * delta_B.grid().mesh().delta[2];
//         dB0 = E(2, i + 1, j + 1) * delta_B.grid().mesh().delta[2] -
//               E(2, i + 1, j) * delta_B.grid().mesh().delta[2];

//         dB0 *= -dt / delta_B.grid().face_area(0, i, j);
//         dB1 *= -dt / delta_B.grid().face_area(1, i, j);
//         dB2 *= -dt / delta_B.grid().face_area(2, i, j);
//         delta_B(0, i, j) = dB0;
//         delta_B(1, i, j) = dB1;
//         delta_B(2, i, j) = dB2;
//       }
//     }
//   } else if (grid.dim() == 3) {
//     // TODO: Add metric coef
//     // 3D, do 3 loops
//     for (int k = 1; k < grid.extent().depth() - 1; k++) {
//       for (int j = 1; j < grid.extent().height() - 1; j++) {
//         for (int i = 1; i < grid.extent().width() - 1; i++) {
//           dB2 = E(1, i + 1, j, k + 1) * delta_B.grid().mesh().delta[1] -
//                 E(1, i, j, k + 1) * delta_B.grid().mesh().delta[1] +
//                 E(0, i, j, k + 1) * delta_B.grid().mesh().delta[0] -
//                 E(0, i, j + 1, k + 1) * delta_B.grid().mesh().delta[0];

//           dB1 = E(0, i, j + 1, k + 1) * delta_B.grid().mesh().delta[0] -
//                 E(0, i, j + 1, k) * delta_B.grid().mesh().delta[0] +
//                 E(2, i, j + 1, k) * delta_B.grid().mesh().delta[2] -
//                 E(2, i + 1, j + 1, k) * delta_B.grid().mesh().delta[2];

//           dB0 = E(2, i + 1, j + 1, k) * delta_B.grid().mesh().delta[2] -
//                 E(2, i + 1, j, k) * delta_B.grid().mesh().delta[2] +
//                 E(1, i + 1, j, k) * delta_B.grid().mesh().delta[1] -
//                 E(1, i + 1, j, k + 1) * delta_B.grid().mesh().delta[1];

//           dB0 *= -dt / delta_B.grid().face_area(0, i, j, k);
//           dB1 *= -dt / delta_B.grid().face_area(1, i, j, k);
//           dB2 *= -dt / delta_B.grid().face_area(2, i, j, k);
//           delta_B(0, i, j, k) = dB0;
//           delta_B(1, i, j, k) = dB1;
//           delta_B(2, i, j, k) = dB2;
//         }
//       }
//     }
//   }  // else do nothing
// }

// void
// FieldSolver_Integral::compute_E_update(vfield_t &delta_E, const vfield_t &B,
//                                        const vfield_t &I, double dt) {
//   // loop over the grid, skipping the outer most index
//   double dE0, dE1, dE2;
//   delta_E.initialize();

//   auto &grid = B.grid();
//   if (grid.dim() == 2) {
//     // 2D, only do 2 loops
//     for (int j = 1; j < grid.extent().height() - 1; j++) {
//       for (int i = 1; i < grid.extent().width() - 1; i++) {
//         dE2 = B(1, i, j - 1) * delta_E.grid().mesh().delta[1] -
//               B(1, i - 1, j - 1) * delta_E.grid().mesh().delta[1] +
//               B(0, i - 1, j - 1) * delta_E.grid().mesh().delta[0] -
//               B(0, i - 1, j) * delta_E.grid().mesh().delta[0] - I(2, i, j);
//         dE1 = B(2, i - 1, j) * delta_E.grid().mesh().delta[2] -
//               B(2, i, j) * delta_E.grid().mesh().delta[2] - I(1, i, j);
//         dE0 = B(2, i, j) * delta_E.grid().mesh().delta[2] -
//               B(2, i, j - 1) * delta_E.grid().mesh().delta[2] - I(0, i, j);

//         dE0 *= dt / delta_E.grid().face_area(0, i, j);
//         dE1 *= dt / delta_E.grid().face_area(1, i, j);
//         dE2 *= dt / delta_E.grid().face_area(2, i, j);
//         delta_E(0, i, j) = dE0;
//         delta_E(1, i, j) = dE1;
//         delta_E(2, i, j) = dE2;
//       }
//     }
//   } else if (grid.dim() == 3) {
//     // TODO: Add metric coef
//     // 3D, do 3 loops
//     for (int k = 1; k < grid.extent().depth() - 1; k++) {
//       for (int j = 1; j < grid.extent().height() - 1; j++) {
//         for (int i = 1; i < grid.extent().width() - 1; i++) {
//           dE2 = B(1, i, j - 1, k) * delta_E.grid().mesh().delta[1] -
//                 B(1, i - 1, j - 1, k) * delta_E.grid().mesh().delta[1] +
//                 B(0, i - 1, j - 1, k) * delta_E.grid().mesh().delta[0] -
//                 B(0, i - 1, j, k) * delta_E.grid().mesh().delta[0] -
//                 I(2, i, j, k);

//           dE1 = B(0, i - 1, j, k) * delta_E.grid().mesh().delta[0] -
//                 B(0, i - 1, j, k - 1) * delta_E.grid().mesh().delta[0] +
//                 B(2, i - 1, j, k - 1) * delta_E.grid().mesh().delta[2] -
//                 B(2, i, j, k - 1) * delta_E.grid().mesh().delta[2] -
//                 I(1, i, j, k);

//           dE0 = B(2, i, j, k - 1) * delta_E.grid().mesh().delta[2] -
//                 B(2, i, j - 1, k - 1) * delta_E.grid().mesh().delta[2] +
//                 B(1, i, j - 1, k - 1) * delta_E.grid().mesh().delta[1] -
//                 B(1, i, j - 1, k) * delta_E.grid().mesh().delta[1] -
//                 I(0, i, j, k);

//           dE0 *= dt / delta_E.grid().face_area(0, i, j, k);
//           dE1 *= dt / delta_E.grid().face_area(1, i, j, k);
//           dE2 *= dt / delta_E.grid().face_area(2, i, j, k);
//           delta_E(0, i, j, k) = dE0;
//           delta_E(1, i, j, k) = dE1;
//           delta_E(2, i, j, k) = dE2;
//         }
//       }
//     }
//   }  // else do nothing
// }

// void
// FieldSolver_Integral::compute_E_update_KS(vfield_t &E, const vfield_t &B,
//                                           const vfield_t &I, double dt) {
//   assert(E.grid().type() == MetricType::Kerr_Schild);
//   std::cout << "Updating Kerr-Schild E field" << std::endl;
//   // Only implemented 2D!
//   if (E.grid().dim() != 2) return;

//   m_dE.assign(0.0);
//   m_ks_rhs.assign(0.0);

//   // Start with D^theta

//   // Construct RHS for D^theta update
//   // TODO: Question of RHS evaluated at 1/2 or 1?
//   auto &mesh = E.grid().mesh();
//   auto &meshB = B.grid().mesh();
//   for (int j = mesh.guard[1]; j < mesh.dims[1] - mesh.guard[1]; j++) {
//     for (int i = mesh.guard[0]; i < mesh.dims[0] - mesh.guard[0]; i++) {
//       // Integral over \phi does not need line cache since everything is
//       // constant over \phi
//       m_ks_rhs(i, j) =
//           E.grid().alpha(1, i, j) * meshB.delta[2] *
//           ((0.5 * (B.grid().metric(0, 2, i - 1, j) * B(0, i - 1, j) +
//                    B.grid().metric(0, 2, i - 2, j) * B(0, i - 2, j)) +
//             B.grid().metric(2, 2, i - 1, j) * B(2, i - 1, j)) -
//            (0.5 * (B.grid().metric(0, 2, i - 1, j) * B(0, i - 1, j) +
//                    B.grid().metric(0, 2, i, j) * B(0, i, j)) +
//             B.grid().metric(2, 2, i, j) * B(2, i, j)));
//       // I rhs
//       m_ks_rhs(i, j) -= I(1, i, j);
//       // D rhs
//       m_ks_rhs(i, j) +=
//           0.5 * meshB.delta[2] * E.grid().det(1, i, j) *
//                            (E.grid().beta2(1, i + 1, j) * E(1, i + 1, j) -
//                             E.grid().beta2(1, i - 1, j) * E(1, i - 1, j));
//       // m_ks_rhs(i, j) += E(1, i, j);
//       m_ks_rhs(i, j) *= dt / m_dE.grid().face_area(1, i, j);
//       m_dE(1, i, j) = m_ks_rhs(i, j) * 0.5;
//       if (j == mesh.guard[1] - 1 || j == mesh.dims[1] - mesh.guard[1])
//         m_dE(1, i, j) = 0.0;
//     }
//   }

//   // Solve the tridiagonal system for D^theta
//   // for (int j = 1; j < mesh.dims[1] - 1; j++) {
//   //   for (int i = 1; i < mesh.dims[0] - 1; i++) {
//   //     // forward sweep
//   //     // m_tri_a[i] = -dt * 0.5 * meshB.delta[2] * E.grid().det(1, i-1, j) *
//   //     // E.grid().beta2(1, i-1, j) / E.grid().face_area(1, i, j);
//   //     m_tri_a[i] = -dt * 0.25 * meshB.delta[2] * E.grid().det(1, i, j) *
//   //                  E.grid().beta2(1, i - 1, j) / E.grid().face_area(1, i, j);
//   //     // m_tri_c[i] = dt * 0.5 * meshB.delta[2] * E.grid().det(1, i+1, j) *
//   //     // E.grid().beta2(1, i+1, j) / E.grid().face_area(1, i, j);
//   //     m_tri_c[i] = dt * 0.25 * meshB.delta[2] * E.grid().det(1, i, j) *
//   //                  E.grid().beta2(1, i + 1, j) / E.grid().face_area(1, i, j);
//   //     m_tri_d[i] = m_ks_rhs(i, j);
//   //     if (i != mesh.guard[0] - 1) {
//   //       m_tri_c[i] = m_tri_c[i] / (1.0 - m_tri_a[i] * m_tri_c[i - 1]);
//   //       m_tri_d[i] = (m_tri_d[i] - m_tri_a[i] * m_tri_d[i - 1]) /
//   //                    (1.0 - m_tri_a[i] * m_tri_c[i - 1]);
//   //     }
//   //   }
//   //   // in the second sweep, we use m_dE to cache the new field result
//   //   for (int i = mesh.dims[0] - 2; i >= 1; i--) {
//   //     if (j == mesh.guard[1] - 1 || j == mesh.dims[1] - mesh.guard[1])
//   //       m_dE(1, i, j) = 0.0;
//   //     else {
//   //       if (i == mesh.dims[0] - 2)
//   //         m_dE(1, i, j) = m_tri_d[i] - E(1, i, j);
//   //       // FIXME: Ugly hack!
//   //       else
//   //         m_dE(1, i, j) = m_tri_d[i] -
//   //                         m_tri_c[i] * (E(1, i + 1, j) + m_dE(1, i + 1, j)) -
//   //                         E(1, i, j);
//   //     }
//   //     // half the change so we could get a time-centered update in the next part
//   //     // m_dE(1, i, j) *= 0.5;
//   //   }
//   // }
//   // m_dE.multiplyBy(0.5);
//   E.addBy(m_dE);
//   // Update finished for D^\theta

//   // TODO: Again, question of whether RHS is evaluated at 1/2 or 1?
//   double dE0;
//   for (int j = mesh.guard[1]; j < mesh.dims[1] - mesh.guard[1]; j++) {
//     for (int i = mesh.guard[0]; i < mesh.dims[0] - mesh.guard[0]; i++) {
//       // Integration over \phi is simply multiplication
//       dE0 = E.grid().alpha(0, i, j) * meshB.delta[2] *
//             ((0.5 * (B.grid().metric(0, 2, i, j) * B(0, i, j) +
//                      B.grid().metric(0, 2, i - 1, j) * B(0, i - 1, j)) +
//               B.grid().metric(2, 2, i, j) * B(2, i, j)) -
//              (0.5 * (B.grid().metric(0, 2, i, j - 1) * B(0, i, j - 1) +
//                      B.grid().metric(0, 2, i - 1, j - 1) * B(0, i - 1, j - 1)) +
//               B.grid().metric(2, 2, i, j - 1) * B(2, i, j - 1)));
//       dE0 -= I(0, i, j);
//       dE0 += 0.5 * E.grid().det(0, i, j) * meshB.delta[2] *
//              (-E.grid().beta2(1, i, j) * E(1, i, j) -
//               E.grid().beta2(1, i + 1, j) * E(1, i + 1, j) +
//               E.grid().beta2(1, i, j - 1) * E(1, i, j - 1) +
//               E.grid().beta2(1, i + 1, j - 1) * E(1, i + 1, j - 1));
//       dE0 *= dt / m_dE.grid().face_area(0, i, j);
//       E(0, i, j) += dE0;
//       // if (dE0 != dE0) {
//       //   std::cout << I(0, i, j) << std::endl;
//       //   std::cout << E.grid().det(0, i, j) << std::endl;
//       //   std::cout << E.grid().beta2(1, i, j) << " " << E.grid().beta2(1, i + 1, j) << std::endl;
//       //   std::cout << E.grid().beta2(1, i, j - 1) << " " << E.grid().beta2(1, i + 1, j - 1) << std::endl;
//       //   std::cout << m_dE.grid().face_area(0, i, j) << std::endl;
//       //   std::cout << "NaN detected! " << dE0 << " at B rhs " << i << " " << j << std::endl;
//       // }
//       // m_dE(0, i, j) = 0.5 * dE0;
//     }
//   }
//   // Add another factor of D^\theta
//   E.addBy(m_dE);

//   // Similar procedure for D^phi
//   for (int j = mesh.guard[1]; j < mesh.dims[1] - mesh.guard[1]; j++) {
//     for (int i = mesh.guard[0]; i < mesh.dims[0] - mesh.guard[0]; i++) {
//       m_ks_rhs(i, j) = B.grid().line_cache(2, i-1, j-1) * 0.5 * (B(2, i-1, j-1) + B(2, i, j-1))
//                        + B.grid().line_cache(0, i-1, j-1) * B(0, i-1, j-1)
//                        - B.grid().line_cache(2, i-1, j) * 0.5 * (B(2, i-1, j) + B(2, i, j))
//                        - B.grid().line_cache(0, i-1, j) * B(0, i-1, j);
//       m_ks_rhs(i, j) += B.grid().line_cache(1, i, j-1) * B(1, i, j-1)
//                         - B.grid().line_cache(1, i-1, j-1) * B(1, i-1, j-1);
//       // m_ks_rhs(i, j) =
//       //     E.grid().alpha(2, i, j) * meshB.delta[0] *
//       //     ((0.5 * (B.grid().metric(2, 0, i, j - 1) * B(2, i, j - 1) +
//       //              B.grid().metric(2, 0, i - 1, j - 1) * B(2, i - 1, j - 1)) +
//       //       B.grid().metric(0, 0, i - 1, j - 1) * B(0, i - 1, j - 1)) -
//       //      (0.5 * (B.grid().metric(0, 2, i - 1, j) * B(0, i - 1, j) +
//       //              B.grid().metric(0, 2, i, j) * B(0, i, j)) +
//       //       B.grid().metric(0, 0, i - 1, j) * B(0, i - 1, j)));
//       // m_ks_rhs(i, j) +=
//       //     E.grid().alpha(2, i, j) * meshB.delta[1] *
//       //     (B.grid().metric(1, 1, i, j - 1) * B(1, i, j - 1) -
//       //      B.grid().metric(1, 1, i - 1, j - 1) * B(1, i - 1, j - 1));
//       // I rhs
//       m_ks_rhs(i, j) -= I(2, i, j);
//       // D rhs
//       m_ks_rhs(i, j) += -E.grid().line_cache(3, i, j) * E(2, i+1, j)
//                         +E.grid().line_cache(3, i-1, j) * E(2, i-1, j);
//       // m_ks_rhs(i, j) +=
//       //     0.5 * meshB.delta[1] * E.grid().det(2, i, j) *
//       //                      (-E.grid().beta1(2, i + 1, j) * E(2, i + 1, j) +
//       //                       E.grid().beta1(2, i - 1, j) * E(2, i - 1, j));
//       // m_ks_rhs(i, j) += E(1, i, j);
//       m_ks_rhs(i, j) *= dt / m_dE.grid().face_area(2, i, j);
//       E(2, i, j) += m_ks_rhs(i, j);
//     }
//   }

//   // TODO: Actually this scheme is not completely integral, since lengths are
//   // not really integrated. Not sure what are the implications

//   // Solve the tridiagonal system for D^phi
//   // for (int j = mesh.guard[1]; j < mesh.dims[1] - mesh.guard[1]; j++) {
//   //   for (int i = mesh.guard[0] - 1; i < mesh.dims[0] - mesh.guard[0] + 1; i++) {
//   //     // forward sweep
//   //     // m_tri_a[i] = -dt * 0.5 * meshB.delta[2] * E.grid().det(1, i-1, j) *
//   //     // E.grid().beta2(1, i-1, j) / E.grid().face_area(1, i, j);
//   //     m_tri_a[i] = dt * 0.25 * meshB.delta[1] * E.grid().det(1, i, j) *
//   //                  E.grid().beta1(2, i - 1, j) / E.grid().face_area(2, i, j);
//   //     // m_tri_c[i] = dt * 0.5 * meshB.delta[2] * E.grid().det(1, i+1, j) *
//   //     // E.grid().beta2(1, i+1, j) / E.grid().face_area(1, i, j);
//   //     m_tri_c[i] = -dt * 0.25 * meshB.delta[1] * E.grid().det(1, i, j) *
//   //                  E.grid().beta1(2, i + 1, j) / E.grid().face_area(2, i, j);
//   //     m_tri_d[i] = m_ks_rhs(i, j);
//   //     if (i != mesh.guard[0] - 1) {
//   //       m_tri_c[i] = m_tri_c[i] / (1.0 - m_tri_a[i] * m_tri_c[i - 1]);
//   //       m_tri_d[i] = (m_tri_d[i] - m_tri_a[i] * m_tri_d[i - 1]) /
//   //                    (1.0 - m_tri_a[i] * m_tri_c[i - 1]);
//   //     }
//   //   }
//   //   // in the second sweep, we use m_dE to cache the new field result
//   //   for (int i = mesh.dims[0] - mesh.guard[0]; i >= mesh.guard[0] - 1; i--) {
//   //     if (i == mesh.dims[0] - mesh.guard[0])
//   //       // m_dE(1, i, j) = m_tri_d[i] - E(1, i, j);
//   //       E(2, i, j) = m_tri_d[i];
//   //     else
//   //       // m_dE(1, i, j) = m_tri_d[i] - m_tri_c[i] * (E(1, i+1, j) + m_dE(1,
//   //       // i+1, j)) - E(1, i, j);
//   //       E(2, i, j) = m_tri_d[i] - m_tri_c[i] * E(2, i + 1, j);
//   //     // half the change so we could get a time-centered update in the next part
//   //     // m_dE(1, i, j) *= 0.5;
//   //   }
//   // }
// }

// void
// FieldSolver_Integral::compute_B_update_KS(vfield_t &B, const vfield_t &E,
//                                           double dt) {
//   assert(B.grid().type() == MetricType::Kerr_Schild);
//   std::cout << "Updating Kerr-Schild B field" << std::endl;
//   // Only implemented 2D!
//   if (B.grid().dim() != 2) return;

//   m_dB.assign(0.0);
//   m_ks_rhs.assign(0.0);

//   // Start with B^theta

//   // Construct RHS for B^theta update
//   // TODO: Question of RHS evaluated at 1/2 or 1?
//   auto &mesh = B.grid().mesh();
//   auto &meshE = E.grid().mesh();
//   for (int j = mesh.guard[1]; j < mesh.dims[1] - mesh.guard[1]; j++) {
//     for (int i = mesh.guard[0]; i < mesh.dims[0] - mesh.guard[0]; i++) {
//       // D rhs
//       m_ks_rhs(i, j) =
//           -B.grid().alpha(1, i, j) * meshE.delta[2] *
//           ((0.5 * (E.grid().metric(0, 2, i, j + 1) * E(0, i, j + 1) +
//                    E.grid().metric(0, 2, i - 1, j + 1) * E(0, i - 1, j + 1)) +
//             E.grid().metric(2, 2, i, j + 1) * E(2, i, j + 1)) -
//            (0.5 * (E.grid().metric(0, 2, i, j + 1) * E(0, i, j + 1) +
//                    E.grid().metric(0, 2, i + 1, j + 1) * E(0, i + 1, j + 1)) +
//             E.grid().metric(2, 2, i + 1, j + 1) * E(2, i + 1, j + 1)));
//       // if (m_ks_rhs(i, j) != m_ks_rhs(i, j)) {
//       //   std::cout << B.grid().alpha(1,i,j) << " " << meshE.delta[2] << std::endl;
//       //   std::cout << E.grid().metric(0, 2, i, j+1) << " " << E.grid().metric(0, 2, i-1, j+1) << std::endl;
//       //   std::cout << E.grid().metric(0, 2, i+1, j+1) << " " << E.grid().metric(2, 2, i+1, j+1) << std::endl;
//       //   std::cout << E.grid().metric(2, 2, i, j+1) << " " << E.grid().metric(2, 2, i+1, j+1) << std::endl;
//       //   std::cout << E(0, i, j+1) << " " << E(0, i-1, j+1) << std::endl;
//       //   std::cout << E(2, i, j+1) << " " << E(2, i+1, j+1) << std::endl;
//       //   std::cout << "NaN detected! " << m_ks_rhs(i, j) << " at D rhs " << i << " " << j << std::endl;
//       //   std::cout << E.grid().metric(2, 2, i, j+1) << " " << E.grid().metric(0, 2, i-1, j+1) << std::endl;
//       // }
//       // B rhs
//       m_ks_rhs(i, j) +=
//           0.5 * meshE.delta[2] * B.grid().det(1, i, j) *
//                            (B.grid().beta2(1, i + 1, j) * B(1, i + 1, j) -
//                             B.grid().beta2(1, i - 1, j) * B(1, i - 1, j));
//       // if (m_ks_rhs(i, j) != m_ks_rhs(i, j)) {
//       //   std::cout << "NaN detected! at B rhs " << i << " " << j << std::endl;
//       // }
//       m_ks_rhs(i, j) *= dt / m_dB.grid().face_area(1, i, j);
//       // if (m_ks_rhs(i, j) != m_ks_rhs(i, j)) {
//       //   std::cout << "NaN detected! at division " << i << " " << j << std::endl;
//       // }
//       m_dB(1, i, j) = m_ks_rhs(i, j) * 0.5;
//     }
//   }

//   // Solve the tridiagonal system for B^theta
//   // for (int j = 1; j < mesh.dims[1] - 1; j++) {
//   //   for (int i = 1; i < mesh.dims[0] - 1; i++) {
//   //     // forward sweep
//   //     // m_tri_a[i] = -dt * 0.5 * meshB.delta[2] * E.grid().det(1, i-1, j) *
//   //     // E.grid().beta2(1, i-1, j) / E.grid().face_area(1, i, j);
//   //     m_tri_a[i] = -dt * 0.25 * meshE.delta[2] * B.grid().det(1, i, j) *
//   //                  B.grid().beta2(1, i - 1, j) / B.grid().face_area(1, i, j);
//   //     // m_tri_c[i] = dt * 0.5 * meshB.delta[2] * E.grid().det(1, i+1, j) *
//   //     // E.grid().beta2(1, i+1, j) / E.grid().face_area(1, i, j);
//   //     m_tri_c[i] = dt * 0.25 * meshE.delta[2] * B.grid().det(1, i, j) *
//   //                  B.grid().beta2(1, i + 1, j) / B.grid().face_area(1, i, j);
//   //     m_tri_d[i] = m_ks_rhs(i, j);
//   //     if (i != 1) {
//   //       m_tri_c[i] = m_tri_c[i] / (1.0 - m_tri_a[i] * m_tri_c[i - 1]);
//   //       m_tri_d[i] = (m_tri_d[i] - m_tri_a[i] * m_tri_d[i - 1]) /
//   //                    (1.0 - m_tri_a[i] * m_tri_c[i - 1]);
//   //     }
//   //   }
//   //   // in the second sweep, we use m_dE to cache the new field result
//   //   for (int i = mesh.dims[0] - 1; i >= 1; i--) {
//   //     if (i == mesh.dims[0] - 1)
//   //       m_dB(1, i, j) = m_tri_d[i] - B(1, i, j);
//   //     else
//   //       m_dB(1, i, j) = m_tri_d[i] -
//   //                       m_tri_c[i] * (B(1, i + 1, j) + m_dB(1, i + 1, j)) -
//   //                       B(1, i, j);
//   //     // half the change so we could get a time-centered update in the next part
//   //   }
//   // }
//   // m_dB.multiplyBy(0.5);
//   B.addBy(m_dB);
//   // Update finished for B^\theta

//   // RHS evaluated at 1/2
//   double dB0;
//   for (int j = mesh.guard[1]; j < mesh.dims[1] - mesh.guard[1]; j++) {
//     for (int i = mesh.guard[0]; i < mesh.dims[0] - mesh.guard[0]; i++) {
//       dB0 = -B.grid().alpha(0, i, j) * meshE.delta[2] *
//             ((0.5 * (E.grid().metric(0, 2, i + 1, j + 1) * E(0, i + 1, j + 1) +
//                      E.grid().metric(0, 2, i, j + 1) * E(0, i, j + 1)) +
//               E.grid().metric(2, 2, i + 1, j + 1) * E(2, i + 1, j + 1)) -
//              (0.5 * (E.grid().metric(0, 2, i + 1, j) * E(0, i + 1, j) +
//                      E.grid().metric(0, 2, i, j) * E(0, i, j)) +
//               E.grid().metric(2, 2, i + 1, j) * E(2, i + 1, j)));
//       // dB0 = -B.grid().alpha(0, i, j) *
//       // if (j == mesh.guard[1] || j == mesh.guard[1] + 1 || j == mesh.guard[1] + 2) {
//       //   std::cout << dB0 << " at " << i << ", " << j << std::endl;
//       // }
//       // if (j == mesh.guard[1]) dB0 *= 2.0r
//       dB0 -= 0.5 * B.grid().det(0, i, j) * meshE.delta[2] *
//             (B.grid().beta2(1, i, j) * B(1, i, j) +
//               B.grid().beta2(1, i + 1, j) * B(1, i + 1, j) -
//               B.grid().beta2(1, i, j - 1) * B(1, i, j - 1) -
//               B.grid().beta2(1, i + 1, j - 1) * B(1, i + 1, j - 1));
//       // }
//       dB0 *= dt / m_dB.grid().face_area(0, i, j);
//       B(0, i, j) += dB0;
//     }
//   }
//   // Add another factor of B^\theta
//   B.addBy(m_dB);

//   // Similar procedure for B^phi
//   for (int j = mesh.guard[1]; j < mesh.dims[1] - mesh.guard[1]; j++) {
//     for (int i = mesh.guard[0]; i < mesh.dims[0] - mesh.guard[0]; i++) {
//       // D rhs
//       m_ks_rhs(i, j) = - E.grid().line_cache(2, i, j) * 0.5 * (E(2, i, j) + E(2, i+1, j))
//                        - E.grid().line_cache(0, i, j) * E(0, i, j)
//                        + E.grid().line_cache(2, i, j+1) * 0.5 * (E(2, i, j+1) + E(2, i+1, j+1))
//                        + E.grid().line_cache(2, i, j+1) * E(0, i, j+1);
//           // -B.grid().alpha(2, i, j) * meshE.delta[0] *
//           // ((0.5 * (E.grid().metric(2, 0, i + 1, j) * E(2, i + 1, j) +
//           //          E.grid().metric(2, 0, i, j) * E(2, i, j)) +
//           //   E.grid().metric(0, 0, i, j) * E(0, i, j)) -
//           //  (0.5 * (E.grid().metric(0, 2, i, j + 1) * E(2, i, j + 1) +
//           //          E.grid().metric(0, 2, i + 1, j + 1) * E(2, i + 1, j + 1)) +
//           //   E.grid().metric(0, 0, i, j + 1) * E(0, i, j + 1)));
//       m_ks_rhs(i, j) += - E.grid().line_cache(1, i+1, j) * E(1, i+1, j)
//                         + E.grid().line_cache(1, i, j) * E(1, i, j);
//       // m_ks_rhs(i, j) += -B.grid().alpha(2, i, j) * meshE.delta[1] *
//       //                   (E.grid().metric(1, 1, i + 1, j) * E(1, i + 1, j) -
//       //                    E.grid().metric(1, 1, i, j) * E(1, i, j));
//       // B rhs
//       m_ks_rhs(i, j) += - B.grid().line_cache(3, i, j) * B(2, i+1, j)
//                         + B.grid().line_cache(3, i-1, j) * B(2, i-1, j);
//       // m_ks_rhs(i, j) +=
//       //     0.5 * meshE.delta[1] * B.grid().det(2, i, j) *
//       //                      (-B.grid().beta1(2, i + 1, j) * B(2, i + 1, j) +
//       //                       B.grid().beta1(2, i - 1, j) * B(2, i - 1, j));
//       m_ks_rhs(i, j) *= dt / m_dB.grid().face_area(2, i, j);
//       // if (m_ks_rhs(i, j) != m_ks_rhs(i, j)) {
//       //   std::cout << "NaN detected! at " << i << " " << j << std::endl;
//       //   std::cout << "Face area 2 here is " << m_dB.grid().face_area(2, i, j) << std::endl;
//       // }
//       B(2, i, j) += m_ks_rhs(i, j);
//     }
//   }

//   // TODO: Actually this scheme is not completely integral, since lengths are
//   // not really integrated. Not sure what are the implications

//   // Solve the tridiagonal system for D^phi
//   // for (int j = 1; j < mesh.dims[1] - 1; j++) {
//   //   for (int i = 1; i < mesh.dims[0] - 1; i++) {
//   //     // forward sweep
//   //     // m_tri_a[i] = -dt * 0.5 * meshB.delta[2] * E.grid().det(1, i-1, j) *
//   //     // E.grid().beta2(1, i-1, j) / E.grid().face_area(1, i, j);
//   //     m_tri_a[i] = dt * 0.25 * meshE.delta[1] * B.grid().det(1, i, j) *
//   //                  B.grid().beta1(2, i - 1, j) / B.grid().face_area(2, i, j);
//   //     // m_tri_c[i] = dt * 0.5 * meshB.delta[2] * E.grid().det(1, i+1, j) *
//   //     // E.grid().beta2(1, i+1, j) / E.grid().face_area(1, i, j);
//   //     m_tri_c[i] = -dt * 0.25 * meshE.delta[1] * B.grid().det(1, i, j) *
//   //                  B.grid().beta1(2, i + 1, j) / B.grid().face_area(2, i, j);
//   //     m_tri_d[i] = m_ks_rhs(i, j);
//   //     if (i != 1) {
//   //       m_tri_c[i] = m_tri_c[i] / (1.0 - m_tri_a[i] * m_tri_c[i - 1]);
//   //       m_tri_d[i] = (m_tri_d[i] - m_tri_a[i] * m_tri_d[i - 1]) /
//   //                    (1.0 - m_tri_a[i] * m_tri_c[i - 1]);
//   //     }
//   //   }
//   //   // in the second sweep, we use m_dE to cache the new field result
//   //   for (int i = mesh.dims[0] - 1; i >= 1; i--) {
//   //     if (i == mesh.dims[0] - 1)
//   //       // m_dE(1, i, j) = m_tri_d[i] - E(1, i, j);
//   //       B(2, i, j) = m_tri_d[i];
//   //     else
//   //       // m_dE(1, i, j) = m_tri_d[i] - m_tri_c[i] * (E(1, i+1, j) + m_dE(1,
//   //       // i+1, j)) - E(1, i, j);
//   //       B(2, i, j) = m_tri_d[i] - m_tri_c[i] * B(2, i + 1, j);
//   //     // half the change so we could get a time-centered update in the next part
//   //     // m_dE(1, i, j) *= 0.5;
//   //   }
//   // }
// }

// void
// FieldSolver_Integral::compute_auxiliary(const vfield_t &E, const vfield_t &B) {
//   auto grid = E.grid();
//   auto mesh = grid.mesh();
//   m_E_aux.assign(0.0);
//   m_H_aux.assign(0.0);
//   // FIXME: Room for optimization: we only need to compute one of E_aux or H_aux
//   // at a time
//   if (grid.dim() == 3) {
//     for (int k = 0; k < grid.extent().depth(); k++) {
//       for (int j = 0; j < grid.extent().height(); j++) {
//         for (int i = 0; i < grid.extent().width(); i++) {
//           int idx = mesh.get_idx(i, j, k);
//           Vec3<int> pos(i, j, k);
//           for (int c = 0; c < 3; c++) {
//             // Take linear combination here to account for non-diagonal metric
//             for (int n = 0; n < 3; n++) {
//               if (grid.metric_mask(c, n))
//                 m_E_aux.data(c)[idx] += E.grid().metric(c, n, idx) *
//                                         E.grid().alpha(c, idx) * E.data(c)[idx];
//               if (B.grid().metric_mask(c, n))
//                 m_H_aux.data(c)[idx] += B.grid().metric(c, n, idx) *
//                                         B.grid().alpha(c, idx) * B.data(c)[idx];
//             }
//             // m_E_aux.data(c)[idx] = E.grid().alpha(c, idx) * E.data(c)[idx];
//             // m_H_aux.data(c)[idx] = B.grid().alpha(c, idx) * B.data(c)[idx];

//             int n_trans[2] = {(c + 1) % 3, (c + 2) % 3};

//             // TODO: Finish the implementation
//             int idx_E_edge = idx - mesh.idx_increment(n_trans[0]) -
//                              mesh.idx_increment(n_trans[1]);
//             if (pos[n_trans[0]] > 0 && pos[n_trans[1]] > 0) {
//               m_E_aux.data(c)[idx] +=
//                   E.grid().beta1(c, idx) *
//                   (B.data(n_trans[1])[idx_E_edge] +
//                    B.data(n_trans[1])[idx_E_edge +
//                                       mesh.idx_increment(n_trans[0])]) *
//                   0.5;
//               m_E_aux.data(c)[idx] -=
//                   E.grid().beta2(c, idx) *
//                   (B.data(n_trans[0])[idx_E_edge] +
//                    B.data(n_trans[0])[idx_E_edge +
//                                       mesh.idx_increment(n_trans[1])]) *
//                   0.5;
//             }

//             int idx_B_edge = idx + mesh.idx_increment(c);
//             if (pos[n_trans[0]] < mesh.dims[n_trans[0]] - 1 &&
//                 pos[c] < mesh.dims[c] - 1)
//               m_H_aux.data(c)[idx] -=
//                   B.grid().beta1(c, idx) *
//                   (E.data(n_trans[1])[idx_B_edge] +
//                    E.data(n_trans[1])[idx_B_edge +
//                                       mesh.idx_increment(n_trans[0])]) *
//                   0.5;
//             if (pos[n_trans[1]] < mesh.dims[n_trans[1]] - 1 &&
//                 pos[c] < mesh.dims[c] - 1)
//               m_H_aux.data(c)[idx] +=
//                   B.grid().beta2(c, idx) *
//                   (E.data(n_trans[0])[idx_B_edge] +
//                    E.data(n_trans[0])[idx_B_edge +
//                                       mesh.idx_increment(n_trans[1])]) *
//                   0.5;
//           }
//         }
//       }
//     }
//   } else if (grid.dim() == 2) {
//     for (int j = 0; j < grid.extent().height(); j++) {
//       for (int i = 0; i < grid.extent().width(); i++) {
//         int idx = mesh.get_idx(i, j, 0);
//         Index pos(i, j, 0);
//         for (int c = 0; c < 3; c++) {
//           // Take linear combination here to account for non-diagonal metric
//           for (int n = 0; n < 3; n++) {
//             if (grid.metric_mask(c, n))
//               m_E_aux.data(c)[idx] += E.grid().metric(c, n, idx) *
//                                       E.grid().alpha(c, idx) * E.data(c)[idx];
//             if (B.grid().metric_mask(c, n))
//               m_H_aux.data(c)[idx] += B.grid().metric(c, n, idx) *
//                                       B.grid().alpha(c, idx) * B.data(c)[idx];
//           }
//           // m_E_aux.data(c)[idx] = E.grid().alpha(c, idx) * E.data(c)[idx] *
//           // E.grid().metric(c, c, idx); m_H_aux.data(c)[idx] =
//           // B.grid().alpha(c, idx) * B.data(c)[idx] * B.grid().metric(c, c,
//           // idx);
//           int n_trans[2] = {(c + 1) % 3, (c + 2) % 3};
//           int inc[3] = {0, 0, 0};
//           // FIXME: Since idxIncrement has built-in check, this is unnecessary?
//           inc[c] = (c == 2 ? 0 : mesh.idx_increment(c));
//           inc[n_trans[0]] =
//               (n_trans[0] == 2 ? 0 : mesh.idx_increment(n_trans[0]));
//           inc[n_trans[1]] =
//               (n_trans[1] == 2 ? 0 : mesh.idx_increment(n_trans[1]));

//           int idx_E_edge = idx - inc[n_trans[0]] - inc[n_trans[1]];

//           // FIXME: Need a better way to detect boundary!!
//           if (E.grid().beta1_mask(c) &&
//               (pos[n_trans[1]] > 0 || inc[n_trans[1]] == 0) &&
//               (pos[n_trans[0]] > 0 || inc[n_trans[0]] == 0))
//             m_E_aux.data(c)[idx] +=
//                 E.grid().det(c, idx) * E.grid().beta1(c, idx) *
//                 (B.data(n_trans[1])[idx_E_edge] +
//                  B.data(n_trans[1])[idx_E_edge + inc[n_trans[0]]]) *
//                 0.5;

//           if (E.grid().beta2_mask(c) &&
//               (pos[n_trans[0]] > 0 || inc[n_trans[0]] == 0) &&
//               (pos[n_trans[1]] > 0 || inc[n_trans[1]] == 0))
//             m_E_aux.data(c)[idx] -=
//                 E.grid().det(c, idx) * E.grid().beta2(c, idx) *
//                 (B.data(n_trans[0])[idx_E_edge] +
//                  B.data(n_trans[0])[idx_E_edge + inc[n_trans[1]]]) *
//                 0.5;

//           int idx_B_edge = idx + inc[c];

//           if (B.grid().beta1_mask(c) && pos[c] < mesh.dims[c] - 1 &&
//               (pos[n_trans[0]] < mesh.dims[n_trans[0]] - 1 ||
//                inc[n_trans[0]] == 0))
//             m_H_aux.data(c)[idx] -=
//                 B.grid().det(c, idx) * B.grid().beta1(c, idx) *
//                 (E.data(n_trans[1])[idx_B_edge] +
//                  E.data(n_trans[1])[idx_B_edge + inc[n_trans[0]]]) *
//                 0.5;

//           if (B.grid().beta2_mask(c) && pos[c] < mesh.dims[c] - 1 &&
//               (pos[n_trans[1]] < mesh.dims[n_trans[1]] - 1 ||
//                inc[n_trans[1]] == 0))
//             // if ((pos[n_trans[1]] < mesh.dims[n_trans[1]] - 1 && pos[c] <
//             // mesh.dims[c] - 1) || inc[n_trans[1]] == 0)
//             m_H_aux.data(c)[idx] +=
//                 B.grid().det(c, idx) * B.grid().beta2(c, idx) *
//                 (E.data(n_trans[0])[idx_B_edge] +
//                  E.data(n_trans[0])[idx_B_edge + inc[n_trans[1]]]) *
//                 0.5;
//         }
//       }
//     }
//   }
// }

// void
// FieldSolver_Integral::compute_GR_auxiliary(const vfield_t &E, const vfield_t &B) {}

// void
// FieldSolver_Integral::compute_flux(const vfield_t &field, sfield_t &flux) {
//   if (field.grid().dim() != 2) {
//     std::cerr << "Need 2D simulation to look at flux!" << std::endl;
//     return;
//   }
//   auto& mesh = field.grid().mesh();
//   for (int i = 0; i < mesh.dims[0]; i++) {
//     flux(i, 0, 0) = 0.0;
//     for (int j = 1; j < mesh.dims[1]; j++) {
//       flux(i, j, 0) = flux(i, j - 1, 0) + field(0, i, j, 0) * field.grid().face_area(0, i, j, 0);
//     }
//   }
// }
