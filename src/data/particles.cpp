#include "data/particles.h"
#include "data/detail/particle_base_impl.hpp"
#include "sim_params.h"

namespace Aperture {

template class ParticleBase<single_particle_t>;
template class ParticleBase<single_photon_t>;

particles_t::particles_t() {}

particles_t::particles_t(std::size_t max_num)
    : ParticleBase<single_particle_t>(max_num) {}

// particles_t::particles_t(const Environment& env, ParticleType type)
particles_t::particles_t(const SimParams& params)
    : ParticleBase<single_particle_t>(
          (std::size_t)params.max_ptc_number) {}

particles_t::particles_t(const particles_t& other)
    : ParticleBase<single_particle_t>(other) {}

particles_t::particles_t(particles_t&& other)
    : ParticleBase<single_particle_t>(std::move(other)) {}

particles_t::~particles_t() {}


}
