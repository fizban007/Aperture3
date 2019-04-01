#include "core/detail/multi_array_utils.hpp"
#include "cuda/constant_mem.h"
#include "cuda/core/additional_diagnostics.h"
#include "cuda/core/cu_sim_data.h"
#include "cuda/core/sim_environment_dev.h"
#include "cuda/cudaUtility.h"
#include "cuda/ptr_util.h"
#include "cuda/utils/iterate_devices.h"

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

AdditionalDiagnostics::AdditionalDiagnostics(
    const cu_sim_environment& env)
    : m_env(env) {
  unsigned int num_devs = env.dev_map().size();
  m_dev_gamma.resize(num_devs);
  m_dev_ptc_num.resize(num_devs);
  for_each_device(env.dev_map(), [this](int n) {
    CudaSafeCall(cudaMallocManaged(
        m_dev_gamma[n],
        sizeof(cudaPitchedPtr) * env.params().num_species));
    CudaSafeCall(cudaMallocManaged(
        m_dev_ptc_num[n],
        sizeof(cudaPitchedPtr) * env.params().num_speciesnum_devs));
  });
}

AdditionalDiagnostics::~AdditionalDiagnostics() {
  for_each_device(m_env.dev_map(), [this](int n) {
    CudaSafeCall(cudaFree(m_dev_gamma[n]));
    CudaSafeCall(cudaFree(m_dev_ptc_num[n]));
  });
}

void
AdditionalDiagnostics::collect_diagnostics(cu_sim_data& data) {
  for_each_device(data.dev_map, [this, &data](int n) {
    data.photon_num[n].initialize();
    data.photon_produced[n].initialize();
    data.pair_produced[n].initialize();
    for (int i = 0; i < data.env.params().num_species; i++) {
      data.gamma[i][n].initialize();
    }
    Kernels::collect_ptc_diagnostics<<<256, 512>>>(
        data.particles[n].data(), data.particles[n].number(),
        m_dev_gamma[n], m_dev_ptc_num[n]);
    CudaCheckError();
    Kernels::collect_photon_diagnostics<<<256, 512>>>(
        data.photons[n].data(), data.photons[n].number(),
        data.photon_num[n].ptr());
    CudaCheckError();
    for (int i = 0; i < m_env.params().num_species; i++) {
      data.gamma[i][n].divideBy(data.photon_num[i][n]);
    }
  });

  cudaDeviceSynchronize();
}

}  // namespace Aperture