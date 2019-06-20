#include "cuda/data/detail/particle_base_impl_dev.hpp"
#include "cuda/data/particles_1d.h"
#include "sim_params.h"

namespace Aperture {

namespace Kernels {

__global__ void
append_ptc(particle1d_data data, size_t num, Pos_t x,
           Scalar p, int cell, ParticleType type, Scalar w,
           uint32_t flag) {
  data.x1[num] = x;
  data.p1[num] = p;
  data.E[num] =
      std::sqrt(1.0f + p * p);
  data.weight[num] = w;
  data.cell[num] = cell;
  data.flag[num] = flag | gen_ptc_type_flag(type);
}


}

template class particle_base_dev<single_particle1d_t>;
// template class particle_base<single_photon1d_t>;

Particles_1D::Particles_1D() {}

Particles_1D::Particles_1D(std::size_t max_num)
    : particle_base_dev<single_particle1d_t>(max_num) {}

Particles_1D::Particles_1D(const SimParams& params)
    : particle_base_dev<single_particle1d_t>(
          (std::size_t)params.max_ptc_number) {}

Particles_1D::Particles_1D(const Particles_1D& other)
    : particle_base_dev<single_particle1d_t>(other) {}

Particles_1D::Particles_1D(Particles_1D&& other)
    : particle_base_dev<single_particle1d_t>(std::move(other)) {}

Particles_1D::~Particles_1D() {}

// void
// Particles_1D::put(std::size_t pos, Pos_t x1, Scalar p1, int cell,
//                   ParticleType type, Scalar weight, uint32_t flag) {
//   if (pos >= m_size)
//     throw std::runtime_error(
//         "Trying to insert particle beyond the end of the array. Resize "
//         "it first!");

//   m_data.x1[pos] = x1;
//   m_data.p1[pos] = p1;
//   m_data.weight[pos] = weight;
//   m_data.cell[pos] = cell;
//   m_data.flag[pos] = flag | gen_ptc_type_flag(type);
//   if (pos >= m_number) m_number = pos + 1;
// }

void
Particles_1D::append(Pos_t x, Scalar p, int cell, ParticleType type,
                     Scalar weight, uint32_t flag) {
  Kernels::append_ptc<<<1, 1>>>(m_data, m_number, x, p, cell, type,
                                weight, flag);
  CudaCheckError();
  m_number += 1;
  cudaDeviceSynchronize();
}

}  // namespace Aperture
