#include "algorithms/functions.h"
#include "algorithms/ptc_pusher_constE.h"
#include "cuda/constant_mem.h"
#include "cuda/cudaUtility.h"
#include "cuda/cuda_control.h"
#include "sim_environment_dev.h"
#include "utils/logger.h"
#include "utils/util_functions.h"

namespace Aperture {

namespace Kernels {

__global__ void
lorentz_push(particle_data ptc, Scalar E, Scalar dt, uint32_t num) {
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num;
       i += blockDim.x * gridDim.x) {
    if (!check_bit(ptc.flag[i], ParticleFlag::ignore_EM)) {
      auto c = ptc.cell[i];
      // Skip empty particles
      if (c == MAX_CELL) continue;
      auto p1 = ptc.p1[i];
      int sp = get_ptc_type(ptc.flag[i]);

      p1 += dev_charges[sp] * dev_params.constE * dt / dev_masses[sp];
      ptc.p1[i] = p1;
    }
  }
}

__global__ void
move_photons(photon_data photons, Scalar dt, uint32_t num) {
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num;
       i += blockDim.x * gridDim.x) {
    auto c = photons.cell[i];
    if (c == MAX_CELL) continue;
    photons.path_left[i] -= dt;
  }
}

}

ParticlePusher_ConstE::ParticlePusher_ConstE(const Environment& env) {
  m_E = env.params().constE;
  // Logger::print_debug("E field is {}", m_E);
}

ParticlePusher_ConstE::~ParticlePusher_ConstE() {}

void
ParticlePusher_ConstE::push(SimData& data, double dt) {
  auto& ptc = data.particles.data();
  auto& photons = data.photons.data();
  Kernels::lorentz_push<<<512, 512>>>(ptc, m_E, dt,
                                      data.particles.number());
  CudaCheckError();

  Kernels::move_photons<<<512, 512>>>(photons, dt,
                                      data.photons.number());
  CudaCheckError();
}

void
ParticlePusher_ConstE::handle_boundary(SimData& data) {
  
}

}