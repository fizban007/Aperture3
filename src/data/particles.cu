#include "data/detail/particle_base_impl.hpp"
#include "data/particles.h"
// #include "sim_environment.h"
#include "sim_params.h"

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

}

template class ParticleBase<single_particle_t>;
template class ParticleBase<single_photon_t>;

Particles::Particles() {}

Particles::Particles(std::size_t max_num)
    : ParticleBase<single_particle_t>(max_num) {}

// Particles::Particles(const Environment& env, ParticleType type)
Particles::Particles(const SimParams& params)
    : ParticleBase<single_particle_t>(
          (std::size_t)params.max_ptc_number) {}

Particles::Particles(const Particles& other)
    : ParticleBase<single_particle_t>(other) {}

Particles::Particles(Particles&& other)
    : ParticleBase<single_particle_t>(std::move(other)) {}

Particles::~Particles() {}

void
Particles::put(std::size_t pos, const Vec3<Pos_t>& x,
               const Vec3<Scalar>& p, int cell, ParticleType type,
               Scalar weight, uint32_t flag) {
  if (pos >= m_numMax)
    throw std::runtime_error(
        "Trying to insert particle beyond the end of the array. Resize "
        "it first!");

  m_data.x1[pos] = x[0];
  m_data.x2[pos] = x[1];
  m_data.x3[pos] = x[2];
  m_data.p1[pos] = p[0];
  m_data.p2[pos] = p[1];
  m_data.p3[pos] = p[2];
  m_data.weight[pos] = weight;
  m_data.cell[pos] = cell;
  m_data.flag[pos] = flag | gen_ptc_type_flag(type);
  if (pos >= m_number) m_number = pos + 1;
}

void
Particles::append(const Vec3<Pos_t>& x, const Vec3<Scalar>& p, int cell,
                  ParticleType type, Scalar weight, uint32_t flag) {
  put(m_number, x, p, cell, type, weight, flag);
}

void
Particles::compute_energies() {
  Kernels::compute_ptc_energies<<<512, 512>>>
      (m_data.p1, m_data.p2, m_data.p3, m_data.E, m_number);
  // Wait for GPU to finish
  cudaDeviceSynchronize();
  CudaCheckError();
}
// void
// Particles::sort(const Grid& grid) {
//   if (m_number > 0)
//     partition_and_sort(m_partition, grid, 8);
// }

}  // namespace Aperture
