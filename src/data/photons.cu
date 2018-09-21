#include "algorithms/functions.h"
#include "cuda/cudaUtility.h"
#include "cuda/cuda_control.h"
#include "data/photons.h"
#include "sim_environment.h"
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

}

Photons::Photons() {}

Photons::Photons(std::size_t max_num)
    : ParticleBase<single_photon_t>(max_num) {}

Photons::Photons(const Environment& env)
    : ParticleBase<single_photon_t>(env.params().max_photon_number) {}

Photons::Photons(const SimParams& params)
    : ParticleBase<single_photon_t>(params.max_photon_number) {}

Photons::~Photons() {}

void
Photons::compute_energies() {
  Kernels::compute_photon_energies<<<512, 512>>>
      (m_data.p1, m_data.p2, m_data.p3, m_data.E, m_number);
  // Wait for GPU to finish
  cudaDeviceSynchronize();
  CudaCheckError();
}

}  // namespace Aperture
