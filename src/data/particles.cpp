#include "data/particles.h"
#include "data/detail/particle_base_impl.hpp"
#include "utils/util_functions.h"
#include "sim_params.h"

namespace Aperture {

template class particle_base<single_particle_t>;
template class particle_base<single_photon_t>;

particles_t::particles_t()
    : particle_base<single_particle_t>() {}

particles_t::particles_t(std::size_t max_num)
    : particle_base<single_particle_t>(max_num) {}

particles_t::particles_t(const SimParams& params)
    : particle_base<single_particle_t>(
          (std::size_t)params.max_ptc_number) {}

particles_t::particles_t(const particles_t& other)
    : particle_base<single_particle_t>(other) {}

particles_t::particles_t(particles_t&& other)
    : particle_base<single_particle_t>(std::move(other)) {}

particles_t::~particles_t() {}

void
particles_t::append(const Vec3<Pos_t>& x, const Vec3<Scalar>& p, int cell,
              ParticleType type, Scalar weight,
                    uint32_t flag) {
  if (m_number >= m_size)
    throw std::runtime_error("Particle array full!");
  m_data.x1[m_number] = x.x;
  m_data.x2[m_number] = x.y;
  m_data.x3[m_number] = x.z;
  m_data.p1[m_number] = p.x;
  m_data.p2[m_number] = p.y;
  m_data.p3[m_number] = p.z;
  m_data.cell[m_number] = cell;
  m_data.E[m_number] = std::sqrt(1.0f + p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
  m_data.weight[m_number] = weight;
  m_data.flag[m_number] = flag | gen_ptc_type_flag(type);

  m_number += 1; 
}


}
