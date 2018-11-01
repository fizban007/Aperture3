#include "data/grid_log_sph.h"

namespace Aperture {

namespace Kernels {}

Grid_LogSph::Grid_LogSph() {}

Grid_LogSph::~Grid_LogSph() {}

void
Grid_LogSph::init(const SimParams& params) {
  Grid::init(params);
  if (m_mesh.dim() == 2) {
    m_h1.resize(m_mesh.dims[0] * 2, m_mesh.dims[1] * 2);
    m_h2.resize(m_mesh.dims[0] * 2, m_mesh.dims[1] * 2);
    m_h3.resize(m_mesh.dims[0] * 2, m_mesh.dims[1] * 2);

    for (int j = 0; j < m_mesh.dims[1] * 2; j++) {
      Scalar x2 = m_mesh.pos(1, j / 2, j % 2);
      for (int i = 0; i < m_mesh.dims[0] * 2; i++) {
        Scalar x1 = m_mesh.pos(0, i / 2, i % 2);
        m_h1(i, j) = std::exp(x1);
        m_h2(i, j) = std::exp(x1);
        m_h3(i, j) = std::exp(x1) * std::sin(x2);
      }
    }
  } else if (m_mesh.dim() == 3) {
    m_h1.resize(m_mesh.dims[0] * 2, m_mesh.dims[1] * 2,
                m_mesh.dims[2] * 2);
    m_h2.resize(m_mesh.dims[0] * 2, m_mesh.dims[1] * 2,
                m_mesh.dims[2] * 2);
    m_h3.resize(m_mesh.dims[0] * 2, m_mesh.dims[1] * 2,
                m_mesh.dims[2] * 2);

    for (int k = 0; k < m_mesh.dims[2] * 2; k++) {
      for (int j = 0; j < m_mesh.dims[1] * 2; j++) {
        Scalar x2 = m_mesh.pos(1, j / 2, j % 2);
        for (int i = 0; i < m_mesh.dims[0] * 2; i++) {
          Scalar x1 = m_mesh.pos(0, i / 2, i % 2);
          m_h1(i, j) = std::exp(x1);
          m_h2(i, j) = std::exp(x1);
          m_h3(i, j) = std::exp(x1) * std::sin(x2);
        }
      }
    }
  }

  m_h1.sync_to_device();
  m_h2.sync_to_device();
  m_h3.sync_to_device();
}

Grid_LogSph::mesh_ptrs
Grid_LogSph::get_mesh_ptrs() const {
  mesh_ptrs ptrs;
  ptrs.h1 = m_h1.data_d();
  ptrs.h2 = m_h2.data_d();
  ptrs.h3 = m_h3.data_d();
  return ptrs;
}

}  // namespace Aperture
