#include "algorithms/functions.h"
#include "cuda/cudaUtility.h"
#include "cuda/cuda_control.h"
#include "data/photons.h"
#include "sim_environment_dev.h"
#include "utils/logger.h"
#include "utils/util_functions.h"

namespace Aperture {

namespace Kernels {

__global__ void
compute_photon_energies(const Scalar* p1, const Scalar* p2,
                        const Scalar* p3, Scalar* E, size_t num) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num;
       i += blockDim.x * gridDim.x) {
    if (i < num) {
      Scalar p1p = p1[i];
      Scalar p2p = p2[i];
      Scalar p3p = p3[i];
      E[i] = std::sqrt(p1p * p1p + p2p * p2p + p3p * p3p);
    }
  }
}

__global__ void
append_ptc(photon_data data, size_t num, Vec3<Pos_t> x, Vec3<Scalar> p,
           Scalar path_left, int cell, Scalar w, uint32_t flag) {
  data.x1[num] = x[0];
  data.x2[num] = x[1];
  data.x3[num] = x[2];
  data.p1[num] = p[0];
  data.p2[num] = p[1];
  data.p3[num] = p[2];
  data.path_left[num] = path_left;
  data.weight[num] = w;
  data.cell[num] = cell;
  data.flag[num] = flag;
}

}  // namespace Kernels

Photons::Photons() {}

Photons::Photons(std::size_t max_num)
    : particle_base_dev<single_photon_t>(max_num) {}

Photons::Photons(const Environment& env)
    : particle_base_dev<single_photon_t>(env.params().max_photon_number) {}

Photons::Photons(const SimParams& params)
    : particle_base_dev<single_photon_t>(params.max_photon_number) {}

Photons::~Photons() {}

void
Photons::compute_energies() {
  Kernels::compute_photon_energies<<<512, 512>>>(
      m_data.p1, m_data.p2, m_data.p3, m_data.E, m_number);
  // Wait for GPU to finish
  cudaDeviceSynchronize();
  CudaCheckError();
}

void
Photons::append(const Vec3<Pos_t>& x, const Vec3<Scalar>& p,
                Scalar path_left, int cell, Scalar weight,
                uint32_t flag) {
  Kernels::append_ptc<<<1, 1>>>(m_data, m_number, x, p, path_left, cell,
                                weight, flag);
  CudaCheckError();
  m_number += 1;
  cudaDeviceSynchronize();
}

}  // namespace Aperture
