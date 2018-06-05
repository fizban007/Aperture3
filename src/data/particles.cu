#include "data/particles.h"
#include "data/detail/particle_base_impl.hpp"
// #include "sim_environment.h"
#include "sim_params.h"

namespace Aperture {
// using boost::fusion::at_c;

template class ParticleBase<single_particle_t>;
template class ParticleBase<single_photon_t>;


Particles::Particles() {}

Particles::Particles(std::size_t max_num, ParticleType type)
    : ParticleBase<single_particle_t>(max_num) {
  m_type = type;

  // TODO: Set charge and mass!!
  double q = 1.0;
  double ion_mass = 1.0;
  if (type == ParticleType::electron) {
    m_charge = -q; m_mass = q;
  } else if (type == ParticleType::positron) {
    m_charge = q; m_mass = q;
  } else if (type == ParticleType::ion) {
    m_charge = q; m_mass = q * ion_mass;
  }
}

// Particles::Particles(const Environment& env, ParticleType type)
Particles::Particles(const SimParams& params, ParticleType type)
    : ParticleBase<single_particle_t>((std::size_t)params.max_ptc_number) {
  m_type = type;
  auto q = params.q_e;
  if (type == ParticleType::electron) {
    m_charge = -q; m_mass = q;
  } else if (type == ParticleType::positron) {
    m_charge = q; m_mass = q;
  } else if (type == ParticleType::ion) {
    m_charge = q; m_mass = q * params.ion_mass;
  }
}

Particles::Particles(const Particles& other)
    : ParticleBase<single_particle_t>(other) {
  m_type = other.m_type;
  m_charge = other.m_charge;
  m_mass = other.m_mass;
}

Particles::Particles(Particles&& other)
    : ParticleBase<single_particle_t>(std::move(other)) {
  m_type = other.m_type;
  m_charge = other.m_charge;
  m_mass = other.m_mass;
}

Particles::~Particles() {}

void
Particles::put(std::size_t pos, Pos_t x, Scalar p, int cell, int flag) {
  if (pos >= m_numMax)
    throw std::runtime_error("Trying to insert particle beyond the end of the array. Resize it first!");

  m_data.x1[pos] = x;
  // m_data.x2[pos] = x[1];
  // m_data.x3[pos] = x[2];
  // m_data.weight[pos] = x[3];
  m_data.p1[pos] = p;
  // m_data.p2[pos] = p[1];
  // m_data.p3[pos] = p[2];
  m_data.gamma[pos] = sqrt(1.0 + p*p);
  m_data.cell[pos] = cell;
  m_data.flag[pos] = flag;
  if (pos >= m_number) m_number = pos + 1;
}

void
Particles::append(Pos_t x, Scalar p, int cell, int flag) {
  put(m_number, x, p, cell, flag);
}

// void
// Particles::sort(const Grid& grid) {
//   if (m_number > 0)
//     partition_and_sort(m_partition, grid, 8);
// }

}
