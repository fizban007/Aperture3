#include "grids/grid_log_sph.h"
#include "omp.h"

namespace Aperture {

Grid_LogSph::Grid_LogSph() {}

Grid_LogSph::~Grid_LogSph() {}

void
Grid_LogSph::compute_coef(const SimParams &params) {
  if (m_mesh.dim() >= 2) {
    // Scalar r_g = params.compactness;
    Scalar r_g = 0.0;
    m_l1_e.resize(m_mesh.dims[0], m_mesh.dims[1]);
    m_l2_e.resize(m_mesh.dims[0], m_mesh.dims[1]);
    m_l3_e.resize(m_mesh.dims[0], m_mesh.dims[1]);
    m_l1_b.resize(m_mesh.dims[0], m_mesh.dims[1]);
    m_l2_b.resize(m_mesh.dims[0], m_mesh.dims[1]);
    m_l3_b.resize(m_mesh.dims[0], m_mesh.dims[1]);

    m_A1_e.resize(m_mesh.dims[0], m_mesh.dims[1]);
    m_A2_e.resize(m_mesh.dims[0], m_mesh.dims[1]);
    m_A3_e.resize(m_mesh.dims[0], m_mesh.dims[1]);
    m_A1_b.resize(m_mesh.dims[0], m_mesh.dims[1]);
    m_A2_b.resize(m_mesh.dims[0], m_mesh.dims[1]);
    m_A3_b.resize(m_mesh.dims[0], m_mesh.dims[1]);

    m_dV.resize(m_mesh.dims[0], m_mesh.dims[1]);

    for (int j = 0; j < m_mesh.dims[1]; j++) {
      double x2 = m_mesh.pos(1, j, 0);
      double x2s = m_mesh.pos(1, j, 1);
      for (int i = 0; i < m_mesh.dims[0]; i++) {
        double x1 = m_mesh.pos(0, i, 0);
        double x1s = m_mesh.pos(0, i, 1);
        double r_plus = std::exp(x1 + m_mesh.delta[0]);
        double r = std::exp(x1);
        double rs = std::exp(x1s);
        double rs_minus = std::exp(x1s - m_mesh.delta[0]);
        m_l1_e(i, j) = l1(rs, r_g) - l1(rs_minus, r_g);
        m_l2_e(i, j) = rs * m_mesh.delta[1];
        m_l3_e(i, j) = rs * std::sin(x2s);
        m_l1_b(i, j) = l1(r_plus, r_g) - l1(r, r_g);
        m_l2_b(i, j) = r * m_mesh.delta[1];
        m_l3_b(i, j) = r * std::sin(x2);

        m_A1_e(i, j) =
            r * r * (std::cos(x2) - std::cos(x2 + m_mesh.delta[1]));
        if (std::abs(x2s) < 1.0e-5) {
          m_A1_e(i, j) =
              r * r * 2.0 * (1.0 - std::cos(0.5 * m_mesh.delta[1]));
        } else if (std::abs(x2s - CONST_PI) < 1.0e-5) {
          m_A1_e(i, j) =
              r * r * 2.0 * (1.0 - std::cos(0.5 * m_mesh.delta[1]));
        }
        m_A2_e(i, j) = (A2(r_plus, r_g) - A2(r, r_g)) * std::sin(x2);
        // Avoid axis singularity
        // if (std::abs(x2s) < 1.0e-5 || std::abs(x2s - CONST_PI)
        // < 1.0e-5)
        //   m_A2_e(i, j) = 0.5 * std::sin(1.0e-5) *
        //                  (std::exp(2.0 * x1s) -
        //                   std::exp(2.0 * (x1s - m_mesh.delta[0])));

        m_A3_e(i, j) = (A2(r_plus, r_g) - A2(r, r_g)) * m_mesh.delta[1];

        m_A1_b(i, j) =
            rs * rs * (std::cos(x2s - m_mesh.delta[1]) - std::cos(x2s));
        m_A2_b(i, j) =
            (A2(rs, r_g) - A2(rs_minus, r_g)) * std::sin(x2s);
        m_A3_b(i, j) =
            (A2(rs, r_g) - A2(rs_minus, r_g)) * m_mesh.delta[1];

        m_dV(i, j) = (V3(r_plus, r_g) - V3(r, r_g)) *
                     (std::cos(x2) - std::cos(x2 + m_mesh.delta[1])) /
                     (m_mesh.delta[0] * m_mesh.delta[1]);

        if (std::abs(x2s) < 1.0e-5 ||
            std::abs(x2s - CONST_PI) < 1.0e-5) {
          m_dV(i, j) = (V3(r_plus, r_g) - V3(r, r_g)) * 2.0 *
                       (1.0 - std::cos(0.5 * m_mesh.delta[1])) /
                       (m_mesh.delta[0] * m_mesh.delta[1]);
          // if (i == 100)
          //   Logger::print_info("dV is {}", m_dV(i, j));
        }
      }
    }
    m_l1_e.copy_to_device();
    m_l2_e.copy_to_device();
    m_l3_e.copy_to_device();
    m_l1_b.copy_to_device();
    m_l2_b.copy_to_device();
    m_l3_b.copy_to_device();

    m_A1_e.copy_to_device();
    m_A2_e.copy_to_device();
    m_A3_e.copy_to_device();
    m_A1_b.copy_to_device();
    m_A2_b.copy_to_device();
    m_A3_b.copy_to_device();

    m_dV.copy_to_device();
  }
}

void
Grid_LogSph::compute_flux(scalar_field<Scalar> &flux,
                          vector_field<Scalar> &B,
                          vector_field<Scalar> &B_bg) const {
  flux.initialize();
  flux.copy_to_host();
  B.copy_to_host();
  auto &mesh = B.grid().mesh();

  for (int j = mesh.guard[1]; j < mesh.dims[1] - mesh.guard[1]; j++) {
    for (int i = mesh.guard[0]; i < mesh.dims[0] - mesh.guard[0]; i++) {
      Scalar r = std::exp(mesh.pos(0, i, true));
      Scalar theta = mesh.pos(1, j, false);
      flux(0, i, j, 0) = flux(0, i, j - 1, 0) +
                         mesh.delta[1] * r * r * std::sin(theta) *
                             (B(0, i, j, 0) + B_bg(0, i, j, 0));
    }
  }
}

}  // namespace Aperture
