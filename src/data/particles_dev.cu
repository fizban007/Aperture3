#include "data/detail/particle_base_impl_dev.hpp"
#include "data/particles_dev.h"
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
  data.E[num] = std::sqrt(1.0f + p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
  data.weight[num] = w;
  data.cell[num] = cell;
  data.flag[num] = flag | gen_ptc_type_flag(type);
}

}  // namespace Kernels

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

// void
// Particles::put(std::size_t pos, const Vec3<Pos_t>& x,
//                const Vec3<Scalar>& p, int cell, ParticleType type,
//                Scalar weight, uint32_t flag) {
//   if (pos >= m_numMax)
//     throw std::runtime_error(
//         "Trying to insert particle beyond the end of the array.
//         Resize " "it first!");

//   m_data.x1[pos] = x[0];
//   m_data.x2[pos] = x[1];
//   m_data.x3[pos] = x[2];
//   m_data.p1[pos] = p[0];
//   m_data.p2[pos] = p[1];
//   m_data.p3[pos] = p[2];
//   m_data.weight[pos] = weight;
//   m_data.cell[pos] = cell;
//   m_data.flag[pos] = flag | gen_ptc_type_flag(type);
//   if (pos >= m_number) m_number = pos + 1;
// }

void
Particles::append(const Vec3<Pos_t>& x, const Vec3<Scalar>& p, int cell,
                  ParticleType type, Scalar weight, uint32_t flag) {
  // put(m_number, x, p, cell, type, weight, flag);
  Kernels::append_ptc<<<1, 1>>>(m_data, m_number, x, p, cell, type, weight, flag);
  CudaCheckError();
  m_number += 1;
  cudaDeviceSynchronize();
}

void
Particles::compute_energies() {
  Kernels::compute_ptc_energies<<<512, 512>>>(
      m_data.p1, m_data.p2, m_data.p3, m_data.E, m_number);
  // Wait for GPU to finish
  cudaDeviceSynchronize();
  CudaCheckError();
}

void
Particles::compute_spectrum(int num_bins, std::vector<Scalar>& energies,
                            std::vector<uint32_t>& nums,
                            ParticleFlag flag) {
  // Assume the particle energies have been computed
  energies.resize(num_bins, 0.0);
  nums.resize(num_bins, 0);

  // Find maximum energy in the array now
  thrust::device_ptr<Scalar> E_ptr =
      thrust::device_pointer_cast(m_data.E);
  Scalar E_max = *thrust::max_element(E_ptr, E_ptr + m_number);
  // Logger::print_info("Maximum energy is {}", E_max);

  // Partition the energy bin up to max energy times a factor
  Scalar dlogE = std::log(E_max) / (Scalar)num_bins;
  for (int i = 0; i < num_bins; i++) {
    energies[i] = std::exp((0.5f + (Scalar)i) * dlogE);
    // Logger::print_info("{}", energies[i]);
  }

  // Do a histogram
  uint32_t* d_energies;
  cudaMalloc(&d_energies, num_bins * sizeof(uint32_t));
  thrust::device_ptr<uint32_t> ptr_energies =
      thrust::device_pointer_cast(d_energies);
  thrust::fill_n(ptr_energies, num_bins, 0);
  cudaDeviceSynchronize();

  compute_energy_histogram(d_energies, m_data.E, m_number, num_bins,
                           E_max, m_data.flag, flag);

  // Copy the resulting histogram to output
  cudaMemcpy(nums.data(), d_energies, num_bins * sizeof(uint32_t),
             cudaMemcpyDeviceToHost);

  cudaFree(d_energies);
}

// void
// Particles::sort(const Grid& grid) {
//   if (m_number > 0)
//     partition_and_sort(m_partition, grid, 8);
// }

}  // namespace Aperture
