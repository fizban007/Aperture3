#include "data/grid_log_sph.h"

namespace Aperture {

namespace Kernels {}

Grid_LogSph::Grid_LogSph() {}

Grid_LogSph::~Grid_LogSph() {}

void
Grid_LogSph::init(const SimParams& params) {
  Grid::init(params);
  if (m_mesh.dim() == 2) {
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

    for (int j = 0; j < m_mesh.dims[1]; j++) {
      Scalar x2 = m_mesh.pos(1, j, 0);
      Scalar x2s = m_mesh.pos(1, j, 1);
      for (int i = 0; i < m_mesh.dims[0]; i++) {
        Scalar x1 = m_mesh.pos(0, i, 0);
        Scalar x1s = m_mesh.pos(0, i, 1);
        m_l1_e(i, j) = std::exp(x1 + m_mesh.delta[0]) - std::exp(x1);
        m_l2_e(i, j) = std::exp(x1) * m_mesh.delta[1];
        m_l3_e(i, j) = std::exp(x1) * std::sin(x2);
        m_l1_b(i, j) = std::exp(x1s) - std::exp(x1s - m_mesh.delta[0]);
        m_l2_b(i, j) = std::exp(x1s) * m_mesh.delta[1];
        m_l3_b(i, j) = std::exp(x1s) * std::sin(x2s);

        m_A1_e(i, j) = std::exp(2.0 * x1s) * (std::cos(x2s - m_mesh.delta[1]) - std::cos(x2s));
        m_A2_e(i, j) = 0.5 * std::sin(x2s) * (std::exp(2.0 * x1s) - std::exp(2.0 * (x1s - m_mesh.delta[0])));
        m_A3_e(i, j) = 0.5 * m_mesh.delta[1] * (std::exp(2.0 * x1s) - std::exp(2.0 * (x1s - m_mesh.delta[0])));
        m_A1_b(i, j) = std::exp(2.0 * x1) * (std::cos(x2) - std::cos(x2 + m_mesh.delta[1]));
        m_A2_b(i, j) = 0.5 * std::sin(x2) * (std::exp(2.0 * (x1 + m_mesh.delta[0])) - std::exp(2.0 * x1));
        m_A3_b(i, j) = 0.5 * m_mesh.delta[1] * (std::exp(2.0 * (x1 + m_mesh.delta[0])) - std::exp(2.0 * x1));
      }
    }
  } else if (m_mesh.dim() == 3) {
    // Do not support 3d yet
  }

  m_l1_e.sync_to_device();
  m_l2_e.sync_to_device();
  m_l3_e.sync_to_device();
  m_l1_b.sync_to_device();
  m_l2_b.sync_to_device();
  m_l3_b.sync_to_device();

  m_A1_e.sync_to_device();
  m_A2_e.sync_to_device();
  m_A3_e.sync_to_device();
  m_A1_b.sync_to_device();
  m_A2_b.sync_to_device();
  m_A3_b.sync_to_device();

}

Grid_LogSph::mesh_ptrs
Grid_LogSph::get_mesh_ptrs() const {
  mesh_ptrs ptrs;
  ptrs.l1_e = m_l1_e.data_d();
  ptrs.l2_e = m_l2_e.data_d();
  ptrs.l3_e = m_l3_e.data_d();
  ptrs.l1_b = m_l1_b.data_d();
  ptrs.l2_b = m_l2_b.data_d();
  ptrs.l3_b = m_l3_b.data_d();

  ptrs.A1_e = m_A1_e.data_d();
  ptrs.A2_e = m_A2_e.data_d();
  ptrs.A3_e = m_A3_e.data_d();
  ptrs.A1_b = m_A1_b.data_d();
  ptrs.A2_b = m_A2_b.data_d();
  ptrs.A3_b = m_A3_b.data_d();

  return ptrs;
}

}  // namespace Aperture
