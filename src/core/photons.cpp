#include "photons.h"

namespace Aperture {

photons_t::photons_t() : base_class() {}

photons_t::photons_t(size_t max_num) : base_class(max_num) {}

photons_t::photons_t(const photons_t& other) : base_class(other) {}

photons_t::photons_t(photons_t&& other)
    : base_class(std::move(other)) {}

photons_t::~photons_t() {}

void
photons_t::append(const Vec3<Pos_t>& x, const Vec3<Scalar>& p, int cell,
                  Scalar path_left, Scalar weight, uint32_t flag) {
  if (m_number >= m_size)
    throw std::runtime_error("Photon array full!");
  m_data.x1[m_number] = x.x;
  m_data.x2[m_number] = x.y;
  m_data.x3[m_number] = x.z;
  m_data.p1[m_number] = p.x;
  m_data.p2[m_number] = p.y;
  m_data.p3[m_number] = p.z;
  m_data.cell[m_number] = cell;
  m_data.path_left[m_number] = path_left;
  m_data.E[m_number] =
      std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
  m_data.weight[m_number] = weight;
  m_data.flag[m_number] = flag;

  m_number += 1;
}

}  // namespace Aperture
