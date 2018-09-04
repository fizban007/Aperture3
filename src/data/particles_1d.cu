#include "data/particles_1d.h"
#include "data/detail/particle_base_impl.hpp"
#include "sim_params.h"

namespace Aperture {

template class ParticleBase<single_particle1d_t>;
template class ParticleBase<single_photon1d_t>;

Particles_1D::Particles_1D() {}

Particles_1D::Particles_1D(std::size_t max_num)
    : ParticleBase<single_particle1d_t>(max_num) {}

Particles_1D::Particles_1D(const SimParams& params)
    : ParticleBase<single_particle1d_t>((std::size_t)params.max_ptc_number) {
}

Particles_1D::Particles_1D(const Particles_1D& other)
    : ParticleBase<single_particle1d_t>(other) {}

Particles_1D::Particles_1D(Particles_1D&& other)
    : ParticleBase<single_particle1d_t>(std::move(other)) {}

Particles_1D::~Particles_1D() {}

void
Particles_1D::put(std::size_t pos, Pos_t x1, Scalar p1, int cell,
                  ParticleType type, Scalar weight, uint32_t flag) {
  if (pos >= m_numMax)
    throw std::runtime_error("Trying to insert particle beyond the end of the array. Resize it first!");

  m_data.x1[pos] = x1;
  m_data.p1[pos] = p1;
  m_data.weight[pos] = weight;
  m_data.cell[pos] = cell;
  m_data.flag[pos] = flag | gen_ptc_type_flag(type);
  if (pos >= m_number) m_number = pos + 1;
}

void
Particles_1D::append(Pos_t x, Scalar p, int cell,
                     ParticleType type, Scalar weight, uint32_t flag) {
  put(m_number, x, p, cell, type, weight, flag);
}

}