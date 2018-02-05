#include "data/photons.h"
#include "sim_environment.h"

namespace Aperture {

Photons::Photons() {}

Photons::Photons(std::size_t max_num)
    : ParticleBase<single_photon_t>(max_num) {}

Photons::Photons(const Environment& env)
    : ParticleBase<single_photon_t>((std::size_t)env.conf().max_photon_number) {}

Photons::Photons(const Photons& other)
    : ParticleBase<single_photon_t>(other) {}

Photons::Photons(Photons&& other)
    : ParticleBase<single_photon_t>(std::move(other)) {}

Photons::~Photons() {}

void
Photons::put(std::size_t pos, Pos_t x, Scalar p, int cell, int flag) {
  if (pos >= m_numMax)
    throw std::runtime_error("Trying to insert photon beyond the end of the array. Resize it first!");

  m_data.x1[pos] = x;
  m_data.p1[pos] = p;
  m_data.cell[pos] = cell;
  m_data.flag[pos] = flag;
  if (pos >= m_number) m_number = pos + 1;
}

void
Photons::append(Pos_t x, Scalar p, int cell, int flag) {
  put(m_number, x, p, cell, flag);
}

void
Photons::convert_pairs(Particles& electrons, Particles& positrons) {}

void
Photons::make_pair(Index_t pos, Particles& electrons, Particles& positrons) {}

}
