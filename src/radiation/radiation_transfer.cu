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

template <typename PtcData, typename PhotonData>
__global__
void count_photon_produced(PtcData ptc, size_t number, int* ph_count,
                           int* phPos, curandState* states) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  CudaRng rng(&states[id]);
  auto inv_comp = make_inverse_compton_PL(dev_params.spectral_alpha, dev_params.e_s,
                                          dev_params.e_min, dev_params.photon_path, rng);
  __shared__ int photonProduced;
  if (threadIdx.x == 0) photonProduced = 0;

  __syncthreads();

  for (uint32_t tid = id; tid < number; tid += blockDim.x * gridDim.x) {
    uint32_t cell = ptc.cell[tid];
    // Skip empty particles
    if (cell == MAX_CELL) continue;

    // TODO: Compute gamma
    Scalar p = ptc.p1[tid];
    Scalar gamma = sqrt(1.0 + p*p);
    if (inv_comp.emit_photon(gamma)) {
      phPos[tid] = atomicAdd(&photonProduced, 1) + 1;
    }
  }

  __syncthreads();

  // Record the number of photons produced this block to global array
  if (threadIdx.x == 0) {
    ph_count[blockIdx.x] = photonProduced;
  }
}

template <typename PtcData, typename PhotonData>
__global__
void produce_photons(PtcData ptc, size_t ptc_num, PhotonData photons, size_t ph_num,
                      int* phPos, int* ph_count, int* ph_cum, curandState* states) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  CudaRng rng(&states[id]);
  auto inv_comp = make_inverse_compton_PL(dev_params.spectral_alpha, dev_params.e_s,
                                          dev_params.e_min, dev_params.photon_path, rng);

  for (uint32_t tid = id; tid < ptc_num; tid += blockDim.x * gridDim.x) {
    int pos_in_block = phPos[tid] - 1;
    if (pos_in_block > -1) {
      int start_pos = ph_cum[blockIdx.x];

      // TODO: Compute gamma
      Scalar p = ptc.p1[tid];
      Scalar gamma = sqrt(1.0 + p*p);
      Scalar Eph = inv_comp.draw_photon_energy(gamma, p);
      Scalar path = inv_comp.draw_photon_freepath(Eph);

      // Add the new photon
      int offset = ph_num + start_pos + pos_in_block;
      photons.x1[offset] = ptc.x1[tid];
      photons.p1[offset] = Eph;
      photons.weight[offset] = ptc.weight[tid];
      photons.path_left[offset] = path;
      photons.cell[offset] = ptc.cell[tid];

      gamma = (gamma - abs(Eph));
      p = sqrt(gamma*gamma - 1);
      ptc.p1[tid] = p;
    }
  }
}

}

template <typename PtcClass, typename PhotonClass>
RadiationTransfer<PtcClass, PhotonClass>::RadiationTransfer(const Environment& env) :
    m_env(env) {
  int seed = m_env.params().random_seed;
  m_threadsPerBlock = 512;
  m_blocksPerGrid = 512;

  CudaSafeCall(cudaMalloc(&d_rand_states, m_threadsPerBlock * m_blocksPerGrid *
                          sizeof(curandState)));
  Kernels::init_rand_states<<<m_blocksPerGrid, m_threadsPerBlock>>>
      ((curandState*)d_rand_states, seed);
  CudaCheckError();

  // Allocate auxiliary arrays for pair creation
  CudaSafeCall(cudaMalloc(&m_numPerBlock, m_blocksPerGrid * sizeof(uint32_t)));
  CudaSafeCall(cudaMalloc(&m_cumNumPerBlock, m_blocksPerGrid * sizeof(uint32_t)));
  CudaSafeCall(cudaMalloc(&m_posInBlock, m_env.params().max_ptc_number));
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