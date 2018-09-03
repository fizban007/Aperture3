#include "radiation/radiation_transfer.h"
#include "cuda/cuda_control.h"
#include "cuda/cudaUtility.h"
#include "cuda/constant_mem.h"
#include "sim_environment.h"
#include "radiation/inverse_compton_power_law.h"
#include <curand_kernel.h>

namespace Aperture {

namespace Kernels {

// Helper struct to plug into inverse compton module
struct CudaRng {
  HOST_DEVICE CudaRng(curandState* state) : m_state(state) {}
  HOST_DEVICE ~CudaRng() {}

  // Generates a device random number between 0.0 and 1.0
  __device__ __forceinline__ float operator()() {
    return curand_uniform(m_state);
  }

  curandState* m_state;
};

__global__
void init_rand_states(curandState* states, int seed) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  curand_init(seed, id, 0, &states[id]);
}

}

template <typename PtcClass, typename PhotonClass>
RadiationTransfer<PtcClass, PhotonClass>::RadiationTransfer(const Environment& env) :
    m_env(env) {
  int seed = env.params().random_seed;
  m_threadsPerBlock = 512;
  m_blocksPerGrid = 512;

  CudaSafeCall(cudaMalloc(&d_rand_states, m_threadsPerBlock * m_blocksPerGrid *
                          sizeof(curandState)));
  Kernels::init_rand_states<<<m_blocksPerGrid, m_threadsPerBlock>>>
      ((curandState*)d_rand_states, seed);
  CudaCheckError();
}

template <typename PtcClass, typename PhotonClass>
RadiationTransfer<PtcClass, PhotonClass>::~RadiationTransfer() {
  cudaFree(d_rand_states);
}

template <typename PtcClass, typename PhotonClass>
void
RadiationTransfer<PtcClass, PhotonClass>::emit_photons(PhotonClass& photons, PtcClass& ptc) {
  
}

template <typename PtcClass, typename PhotonClass>
void
RadiationTransfer<PtcClass, PhotonClass>::produce_pairs(PtcClass& ptc, PhotonClass& photons) {

}

////////////////////////////////////////////////////////////////////////////////
//  Explicit instantiations
////////////////////////////////////////////////////////////////////////////////

template class RadiationTransfer<particle1d_data, photon1d_data>;
template class RadiationTransfer<particle_data, photon_data>;

}