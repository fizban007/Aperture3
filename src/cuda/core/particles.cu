#include "core/particles.h"
#include "particle_base_impl.cuh"

namespace Aperture {

namespace Kernels {

__global__ void
compute_ptc_energies(const Scalar* p1, const Scalar* p2,
                     const Scalar* p3, Scalar* E, size_t num) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num;
       i += blockDim.x * gridDim.x) {
    if (i < num) {
      Scalar p1p = p1[i];
      Scalar p2p = p2[i];
      Scalar p3p = p3[i];
      E[i] = std::sqrt(1.0f + p1p * p1p + p2p * p2p + p3p * p3p);
    }
  }
}

__global__ void
append_ptc(particle_data data, size_t num, Vec3<Pos_t> x,
           Vec3<Scalar> p, int cell, ParticleType type, Scalar w,
           uint32_t flag) {
  data.x1[num] = x[0];
  data.x2[num] = x[1];
  data.x3[num] = x[2];
  data.p1[num] = p[0];
  data.p2[num] = p[1];
  data.p3[num] = p[2];
  data.E[num] =
      std::sqrt(1.0f + p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
  data.weight[num] = w;
  data.cell[num] = cell;
  data.flag[num] = flag | gen_ptc_type_flag(type);
}

}  // namespace Kernels

template class particle_base<single_particle_t>;
template class particle_base<single_photon_t>;

particles_t::particles_t()
    : particle_base<single_particle_t>() {}

particles_t::particles_t(std::size_t max_num)
    : particle_base<single_particle_t>(max_num) {}

particles_t::particles_t(const particles_t& other)
    : particle_base<single_particle_t>(other) {}

particles_t::particles_t(particles_t&& other)
    : particle_base<single_particle_t>(std::move(other)) {}

particles_t::~particles_t() {}

void
particles_t::append(const Vec3<Pos_t>& x, const Vec3<Scalar>& p,
                    int cell, ParticleType type, Scalar weight,
                    uint32_t flag) {
  if (m_number >= m_size)
    throw std::runtime_error("Particle array full!");
  Kernels::append_ptc<<<1, 1>>>(m_data, m_number, x, p, cell, type,
                                weight, flag);
  CudaCheckError();
  m_number += 1;
  cudaDeviceSynchronize();
}

}  // namespace Aperture
