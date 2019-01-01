#include "additional_diagnostics.h"
#include "cuda/constant_mem.h"
#include "cuda/cudaUtility.h"
#include "cuda/ptr_util.h"
#include "data/detail/multi_array_utils.hpp"
#include "sim_data_dev.h"
#include "sim_environment_dev.h"

namespace Aperture {

namespace Kernels {

__global__ void
collect_ptc_diagnostics(particle_data ptc, size_t num,
                        cudaPitchedPtr* ptc_gamma,
                        cudaPitchedPtr* ptc_num) {
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num;
       idx += blockDim.x * gridDim.x) {
    auto c = ptc.cell[idx];
    // Skip empty particles
    if (c == MAX_CELL) continue;

    // Load particle quantities
    int c1 = dev_mesh.get_c1(c);
    int c2 = dev_mesh.get_c2(c);
    // auto p1 = ptc.p1[idx], p2 = ptc.p2[idx], p3 = ptc.p3[idx];
    auto gamma = ptc.E[idx];
    auto flag = ptc.flag[idx];
    int sp = get_ptc_type(flag);
    auto w = ptc.weight[idx];
    size_t offset = c2 * ptc_num[0].pitch + c1 * sizeof(Scalar);

    // printf("%d\n",ptc_num[0].pitch);
    atomicAdd(ptrAddr(ptc_num[sp], offset), w);
    atomicAdd(ptrAddr(ptc_gamma[sp], offset), w * gamma);
  }
}

__global__ void
collect_photon_diagnostics(photon_data photons, size_t num,
                           cudaPitchedPtr photon_num) {
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num;
       idx += blockDim.x * gridDim.x) {
    auto c = photons.cell[idx];
    // Skip empty particles
    if (c == MAX_CELL) continue;

    // Load particle quantities
    int c1 = dev_mesh.get_c1(c);
    int c2 = dev_mesh.get_c2(c);
    auto w = photons.weight[idx];
    // auto p1 = ptc.p1[idx], p2 = ptc.p2[idx], p3 = ptc.p3[idx];
    size_t offset = c2 * photon_num.pitch + c1 * sizeof(Scalar);

    atomicAdd(ptrAddr(photon_num, offset), w);
  }
}

}  // namespace Kernels

AdditionalDiagnostics::AdditionalDiagnostics(const Environment& env)
    : m_env(env), m_ph_num(env.local_grid()) {
  CudaSafeCall(cudaMallocManaged(
      &m_dev_gamma,
      m_env.params().num_species * sizeof(cudaPitchedPtr)));
  CudaSafeCall(cudaMallocManaged(
      &m_dev_ptc_num,
      m_env.params().num_species * sizeof(cudaPitchedPtr)));

  for (int i = 0; i < m_env.params().num_species; i++) {
    m_gamma.emplace_back(m_env.local_grid());
    m_ptc_num.emplace_back(m_env.local_grid());
  }
  for (int i = 0; i < m_env.params().num_species; i++) {
    m_dev_gamma[i] = m_gamma[i].ptr();
    m_dev_ptc_num[i] = m_ptc_num[i].ptr();
  }
}

AdditionalDiagnostics::~AdditionalDiagnostics() {
  CudaSafeCall(cudaFree(m_dev_gamma));
  CudaSafeCall(cudaFree(m_dev_ptc_num));
}

void
AdditionalDiagnostics::collect_diagnostics(const SimData& data) {
  m_ph_num.initialize();
  for (int i = 0; i < m_env.params().num_species; i++) {
    m_gamma[i].initialize();
    m_ptc_num[i].initialize();
  }
  Kernels::collect_ptc_diagnostics<<<256, 512>>>(
      data.particles.data(), data.particles.number(), m_dev_gamma,
      m_dev_ptc_num);
  CudaCheckError();

  Kernels::collect_photon_diagnostics<<<256, 512>>>(
      data.photons.data(), data.photons.number(), m_ph_num.ptr());
  CudaCheckError();

  for (int i = 0; i < m_env.params().num_species; i++) {
    m_gamma[i].divideBy(m_ptc_num[i]);
  }
}

}  // namespace Aperture