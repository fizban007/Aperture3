#include "data/detail/particle_base_impl_dev.hpp"
#include "data/photons_1d.h"
#include "sim_params.h"

namespace Aperture {

Photons_1D::Photons_1D() {}

Photons_1D::Photons_1D(std::size_t max_num)
    : particle_base<single_photon1d_t>(max_num) {}

Photons_1D::Photons_1D(const SimParams& params)
    : particle_base<single_photon1d_t>(
          (std::size_t)params.max_ptc_number) {}

Photons_1D::Photons_1D(const Photons_1D& other)
    : particle_base<single_photon1d_t>(other) {}

Photons_1D::Photons_1D(Photons_1D&& other)
    : particle_base<single_photon1d_t>(std::move(other)) {}

Photons_1D::~Photons_1D() {}

void
Photons_1D::put(std::size_t pos, Pos_t x1, Scalar p1, Scalar path_left,
                int cell, Scalar weight, uint32_t flag) {
  if (pos >= m_numMax)
    throw std::runtime_error(
        "Trying to insert particle beyond the end of the array. Resize "
        "it first!");

  m_data.x1[pos] = x1;
  m_data.p1[pos] = p1;
  m_data.path_left[pos] = path_left;
  m_data.weight[pos] = weight;
  m_data.cell[pos] = cell;
  m_data.flag[pos] = flag;
  if (pos >= m_number) m_number = pos + 1;
}

void
Photons_1D::append(Pos_t x1, Scalar p1, Scalar path_left, int cell,
                   Scalar weight, uint32_t flag) {
  put(m_number, x1, p1, path_left, cell, weight, flag);
}

}  // namespace Aperture