#include "algorithms/current_deposit_Esirkepov.h"
#include "algorithms/interpolation.h"
#include "utils/util_functions.h"

using namespace Aperture;

CurrentDepositer_Esirkepov::CurrentDepositer_Esirkepov(const Environment& env)
    : m_env(env) {}

CurrentDepositer_Esirkepov::~CurrentDepositer_Esirkepov() {}

void CurrentDepositer_Esirkepov::deposit(SimData& data, double dt) {
  Logger::print_detail("Depositing current");
  auto& part = data.particles;
  data.J.initialize();

  for (Index_t i = 0; i < part.size(); i++) {
    data.Rho[i].initialize();
    split_delta_rho(data.J, data.Rho[i], part[i], dt);
    normalize_density(data.Rho[i], data.Rho[i]);
  }

  // communication on the just deposited Rho
  if (m_comm_rho != nullptr) {
    for (Index_t i = 0; i < part.size(); i++) {
      m_comm_rho(data.Rho[i]);
    }
  }
  // Now we have delta Q in every cell, add them up along all directions

  scan_current(data.J);
  // Call communication on just scanned J
  if (m_comm_J != nullptr) {
    m_comm_J(data.J);
  }
}

void CurrentDepositer_Esirkepov::split_delta_rho(vfield& J, sfield& Rho,
                                                 const Particles& particles,
                                                 double dt) {
  Interpolator interp(m_deposit_order);
  auto& part = particles.data();
  auto& grid = J.grid();
  auto charge = particles.charge();
  if (grid.dim() == 1) {
    Logger::print_debug("Computing rho");
    // loop over all the particles
    for (Index_t n = 0; n < particles.number(); n++) {
      if (particles.is_empty(n)) continue;

      int c = part.cell[n];
      int c_p = c;
      auto x = part.x1[n];
      auto x_p = part.x1[n] - part.dx1[n];
      c_p += std::floor(x_p);
      x_p -= c_p - c;
      // auto v3 = part.dx3[n];

      for (int i = c_p - interp.radius();
           i <= c_p + interp.support() - interp.radius(); i++) {
        double w0 = interp.interp_cell(x, c, i);
        // double w1 = interp.interp_cell(x[1], c[1], j);
        double s1 = w0;
        // This is kinda redundant
        int idx = grid.mesh().get_idx(i);

        if (!check_bit(part.flag[n], ParticleFlag::ignore_current)) {
          double w0_p = interp.interp_cell(x_p, c_p, i);
          // double w1_p = interp.interp_cell(x_p[1], c_p[1], j);
          double s0 = w0_p;
          // double s00 = w0_p * w1_p;
          J.data(0)[idx] += charge * (s1 - s0) / dt;
          // TODO: figure out J1 and J2


          // J.data(1)[idx] += charge * 0.5 * (s11 - s10 + s01 - s00) / dt;
          // J.data(2)[idx] += grid.face_area(2, idx) * charge * v3 *
          //                   (s11 + 0.5 * s10 + 0.5 * s01 + s00) /
          //                   (3.0 * grid.cell_volume(idx) * dt);
        }
        Rho.data()[idx] += charge * s1;
        // std::cout << idx << ": " << J.data(0)[idx] << ", " <<
        // Rho.data()[idx] << std::endl;
        // Logger::print_debug("weights are {}, {}", w0, w1);
        // Logger::print_debug("{} : {}, {}, {}, and {}", idx, J.data(0)[idx], J.data(1)[idx], J.data(2)[idx], Rho.data()[idx]);
      }
    }
  }
  // if (grid.dim() == 2) {
  //   Logger::print_debug("Computing rho");
  //   // loop over all the particles
  //   for (Index_t n = 0; n < particles.number(); n++) {
  //     if (particles.is_empty(n)) continue;
  //     double v3 = part.dx3[n];
  //     // std::cout << "v3 is " << v3 << ", h3 is " << std::sqrt(grid.metric(2,
  //     // 2, part.cell[n])) << std::endl; std::cout << "A3 / dV is " <<
  //     // grid.face_area(2, part.cell[n]) / grid.cell_volume(part.cell[n]) <<
  //     // std::endl;

  //     Vec3<int> c = grid.mesh().get_cell_3d(part.cell[n]);
  //     // This is after movement so we need to figure out the original
  //     // position of the particle
  //     auto x = Vec3<float>(part.x1[n], part.x2[n], part.x3[n]);
  //     auto x_p = x - Vec3<float>(part.dx1[n], part.dx2[n],
  //                                part.dx3[n]);  // previous relative position
  //     // auto& x = part.x[n];
  //     // Vec3<float> x_p = (x - part.dx[n]).vec3(); // previous relative
  //     // position
  //     Vec3<int> c_p = c;
  //     c_p[0] += std::floor(x_p[0]);
  //     c_p[1] += std::floor(x_p[1]);
  //     x_p[0] -= c_p[0] - c[0];
  //     x_p[1] -= c_p[1] - c[1];

  //     for (int j = c_p[1] - interp.radius();
  //          j <= c_p[1] + interp.support() - interp.radius(); j++) {
  //       for (int i = c_p[0] - interp.radius();
  //            i <= c_p[0] + interp.support() - interp.radius(); i++) {
  //         double w0 = interp.interp_cell(x[0], c[0], i);
  //         double w1 = interp.interp_cell(x[1], c[1], j);
  //         double s11 = w0 * w1;
  //         int idx = grid.mesh().get_idx(i, j);

  //         // if ((part.flag[n] & ParticleFlag::ignore_current) != 1) {
  //         if (!check_bit(part.flag[n], ParticleFlag::ignore_current)) {
  //           double w0_p = interp.interp_cell(x_p[0], c_p[0], i);
  //           double w1_p = interp.interp_cell(x_p[1], c_p[1], j);
  //           double s10 = w0 * w1_p;
  //           double s01 = w0_p * w1;
  //           double s00 = w0_p * w1_p;
  //           J.data(0)[idx] += charge * 0.5 * (s11 - s01 + s10 - s00) / dt;
  //           J.data(1)[idx] += charge * 0.5 * (s11 - s10 + s01 - s00) / dt;
  //           J.data(2)[idx] += grid.face_area(2, idx) * charge * v3 *
  //                             (s11 + 0.5 * s10 + 0.5 * s01 + s00) /
  //                             (3.0 * grid.cell_volume(idx) * dt);
  //         }
  //         Rho.data()[idx] += charge * s11;
  //         // std::cout << idx << ": " << J.data(0)[idx] << ", " <<
  //         // Rho.data()[idx] << std::endl;
  //         // Logger::print_debug("weights are {}, {}", w0, w1);
  //         // Logger::print_debug("{} : {}, {}, {}, and {}", idx, J.data(0)[idx], J.data(1)[idx], J.data(2)[idx], Rho.data()[idx]);
  //       }
  //     }

  //     // std::cout << J.data(2)[part.cell[n]] / grid.face_area(2, part.cell[n])
  //     // << ", " << Rho.data()[part.cell[n]] / grid.cell_volume(part.cell[n]) <<
  //     // std::endl;
  //   }
  // }  // else do nothing.
}

void CurrentDepositer_Esirkepov::scan_current(vfield& J) {
  auto& grid = J.grid();
  if (grid.dim() == 3) {
    for (int dir = 0; dir < 3; dir++) {
      auto& J_data = J.data(dir);
      int trans[2] = {(dir + 1) % 3, (dir + 2) % 3};

      int inc_trans[2] = {grid.mesh().idx_increment(trans[0]),
                          grid.mesh().idx_increment(trans[1])};
      int inc_dir = grid.mesh().idx_increment(dir);
      for (int k = grid.mesh().guard[trans[0]];
           k < grid.mesh().dims[trans[0]] - grid.mesh().guard[trans[0]]; k++) {
        for (int j = grid.mesh().guard[trans[1]];
             j < grid.mesh().dims[trans[1]] - grid.mesh().guard[trans[1]]; j++) {
          int transIdx = k * inc_trans[0] + j * inc_trans[1];
          for (int i = 1; i < grid.mesh().dims[dir]; i++) {
            J_data[i * inc_dir + transIdx] +=
                J_data[(i - 1) * inc_dir + transIdx];
          }
        }
      }
    }
  } else if (grid.dim() == 2) {
    Logger::print_detail("Scanning current");
    for (int j = grid.mesh().guard[1]; j < grid.mesh().dims[1] - grid.mesh().guard[1]; j++) {
      for (int i = 1; i < grid.mesh().dims[0]; i++) {
        J.data(0)[i + j*grid.mesh().idx_increment(1)] += J.data(0)[i-1 + j*grid.mesh().idx_increment(1)];
      }
    }
    for (int i = grid.mesh().guard[0]; i < grid.mesh().dims[0] - grid.mesh().guard[0]; i++) {
      for (int j = 1; j < grid.mesh().dims[1]; j++) {
        J.data(1)[i + j*grid.mesh().idx_increment(1)] += J.data(1)[i + (j-1)*grid.mesh().idx_increment(1)];
        // if (J(1, i, j) != 0.0)
        //   Logger::print_debug("J here is {}", J(1, i, j));
      }
    }
  } else if (grid.dim() == 1) {
    for (int i = 1; i < grid.mesh().dims[0]; i++) {
      J.data(0)[i] += J.data(0)[i-1];
    }
  }
}

// FIXME: Boundary conditions!
void CurrentDepositer_Esirkepov::normalize_current(const vfield& I, vfield& J) {
  auto& grid = I.grid();
  auto& mesh = I.grid().mesh();
  if (grid.dim() == 1) {
    for (int i = 0; i < mesh.dims[0]; i++) {
      J.data(0)[i] = I.data(0)[i] * mesh.delta[0];
    }
  }
  // for (int k = 0; k < mesh.dims[2]; k++) {
  //   int idx_k = k * mesh.idx_increment(2);
  //   for (int j = 0; j < mesh.dims[1]; j++) {
  //     int idx_j = j * mesh.idx_increment(1);
  //     for (int i = 0; i < mesh.dims[0]; i++) {
  //       int idx = i + idx_j + idx_k;
  //       for (int comp = 0; comp < 3; comp++) {
  //         if (grid.face_area(comp, idx) > 1.0e-6)
  //           J.data(comp)[idx] =
  //               std::sqrt(grid.metric(comp, comp, idx)) * I.data(comp)[idx] /
  //               (grid.face_area(comp, idx) * grid.alpha(comp, idx));
  //         else
  //           J.data(comp)[idx] = 0.0;
  //       }
  //     }
  //   }
  // }
}

void CurrentDepositer_Esirkepov::normalize_density(const sfield& Q,
                                                   sfield& rho) {
}
//   auto& grid = Q.grid();
//   auto& mesh = Q.grid().mesh();
//   for (int k = 0; k < mesh.dims[2]; k++) {
//     int idx_k = k * mesh.idx_increment(2);
//     for (int j = 0; j < mesh.dims[1]; j++) {
//       int idx_j = j * mesh.idx_increment(1);
//       for (int i = 0; i < mesh.dims[0]; i++) {
//         int idx = i + idx_j + idx_k;
//         rho.data()[idx] = Q.data()[idx] / grid.cell_volume(idx);
//       }
//     }
//   }
// }
