#include "cuda/constant_mem.h"
#include "cuda/cudaUtility.h"
#include "cuda/cuda_control.h"
#include "cuda/cudarng.h"
#include "data/particles.h"
#include "data/particles_1d.h"
#include "data/photons.h"
#include "data/photons_1d.h"
#include "radiation/inverse_compton_dummy.h"
#include "radiation/inverse_compton_power_law.h"
#include "radiation/radiation_transfer.h"
#include "sim_environment.h"
#include "utils/logger.h"
#include "utils/util_functions.h"
#include <curand_kernel.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

namespace Aperture {

namespace Kernels {

__global__ void
init_rand_states(curandState* states, int seed) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  curand_init(seed, id, 0, &states[id]);
}

template <typename PtcData, typename RadModel>
__global__ void
count_photon_produced(PtcData ptc, size_t number, int* ph_count,
                      int* phPos, curandState* states) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  CudaRng rng(&states[id]);
  // auto inv_comp = make_inverse_compton_PL(dev_params.spectral_alpha,
  // dev_params.e_s,
  //                                         dev_params.e_min,
  //                                         dev_params.photon_path,
  //                                         rng);
  RadModel rad_model(dev_params, rng);
  // auto inv_comp = make_inverse_compton_dummy(10.0, )
  __shared__ int photonProduced;
  if (threadIdx.x == 0) photonProduced = 0;

  __syncthreads();

  for (uint32_t tid = id; tid < number; tid += blockDim.x * gridDim.x) {
    uint32_t cell = ptc.cell[tid];
    // Skip empty particles
    if (cell == MAX_CELL) continue;

    Scalar p = ptc.p1[tid];
    Scalar gamma = sqrt(1.0 + p * p);
    if (rad_model.emit_photon(gamma)) {
      phPos[tid] = atomicAdd(&photonProduced, 1) + 1;
    }
  }

  __syncthreads();

  // Record the number of photons produced this block to global array
  if (threadIdx.x == 0) {
    ph_count[blockIdx.x] = photonProduced;
  }
}

template <typename PtcData, typename PhotonData, typename RadModel>
__global__ void
produce_photons(PtcData ptc, size_t ptc_num, PhotonData photons,
                size_t ph_num, int* phPos, int* ph_count, int* ph_cum,
                curandState* states) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  CudaRng rng(&states[id]);
  // auto inv_comp = make_inverse_compton_PL(dev_params.spectral_alpha,
  // dev_params.e_s,
  //                                         dev_params.e_min,
  //                                         dev_params.photon_path,
  //                                         rng);
  RadModel rad_model(dev_params, rng);
  for (uint32_t tid = id; tid < ptc_num;
       tid += blockDim.x * gridDim.x) {
    int pos_in_block = phPos[tid] - 1;
    if (pos_in_block > -1) {
      int start_pos = ph_cum[blockIdx.x];

      // TODO: Compute gamma
      Scalar p = ptc.p1[tid];
      Scalar gamma = sqrt(1.0 + p * p);
      Scalar Eph = rad_model.draw_photon_energy(gamma, p);
      gamma = (gamma - std::abs(Eph));
      p = sgn(p) * sqrt(gamma * gamma - 1);
      ptc.p1[tid] = p;

      // If photon energy is too low, do not track it, but still
      // subtract its energy as done above
      if (std::abs(Eph) < dev_params.E_ph_min) continue;

      // Add the new photon
      Scalar path = rad_model.draw_photon_freepath(Eph);
      if (path > dev_params.lph_cutoff) continue;
      // printf("Eph is %f, path is %f\n", Eph, path);
      int offset = ph_num + start_pos + pos_in_block;
      photons.x1[offset] = ptc.x1[tid];
      photons.p1[offset] = Eph;
      photons.weight[offset] = ptc.weight[tid];
      photons.path_left[offset] = path;
      photons.cell[offset] = ptc.cell[tid];
    }
  }
}

template <typename PhotonData>
__global__ void
count_pairs_produced(PhotonData photons, size_t number, int* pair_count,
                     int* pair_pos, curandState* states) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  // CudaRng rng(&states[id]);
  // auto inv_comp = make_inverse_compton_PL(dev_params.spectral_alpha,
  // dev_params.e_s,
  //                                         dev_params.e_min,
  //                                         dev_params.photon_path,
  //                                         rng);
  __shared__ int pairsProduced;
  if (threadIdx.x == 0) pairsProduced = 0;

  __syncthreads();

  for (uint32_t tid = id; tid < number; tid += blockDim.x * gridDim.x) {
    // if (tid >= number) continue;
    uint32_t cell = photons.cell[tid];
    // Skip empty photons
    if (cell == MAX_CELL) continue;

    if (photons.path_left[tid] <= 0.0f) {
      pair_pos[tid] = atomicAdd(&pairsProduced, 1) + 1;
    }
  }

  __syncthreads();

  // Record the number of pairs produced this block to global array
  if (threadIdx.x == 0) {
    pair_count[blockIdx.x] = pairsProduced;
  }
}

template <typename PtcData, typename PhotonData>
__global__ void
produce_pairs(PhotonData photons, size_t ph_num, PtcData ptc,
              size_t ptc_num, int* pair_pos, int* pair_count,
              int* pair_cum, curandState* states) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  // CudaRng rng(&states[id]);
  // auto inv_comp = make_inverse_compton_PL1D(dev_params,
  // dev_params.photon_path, rng); RadModel rad_model(dev_params, rng);

  for (uint32_t tid = id; tid < ph_num;
       tid += blockDim.x * gridDim.x) {
    int pos_in_block = pair_pos[tid] - 1;
    if (pos_in_block > -1 && photons.cell[tid] != MAX_CELL) {
      int start_pos = pair_cum[blockIdx.x] * 2;

      // Split the photon energy evenly between the pairs
      Scalar E_ph = photons.p1[tid];
      Scalar new_p = std::sqrt(max(0.25f * E_ph * E_ph, 1.0f) - 1.0f);

      // Add the two new particles
      int offset = ptc_num + start_pos + pos_in_block * 2;
      ptc.x1[offset] = photons.x1[tid];
      ptc.x1[offset + 1] = photons.x1[tid];
      ptc.p1[offset] = sgn(E_ph) * new_p;
      ptc.p1[offset + 1] = sgn(E_ph) * new_p;
      ptc.weight[offset] = photons.weight[tid];
      ptc.weight[offset + 1] = photons.weight[tid];
      ptc.cell[offset] = photons.cell[tid];
      ptc.cell[offset + 1] = photons.cell[tid];
      ptc.flag[offset] = set_ptc_type_flag(
          (uint32_t)ParticleFlag::secondary, ParticleType::electron);
      ptc.flag[offset + 1] = set_ptc_type_flag(
          (uint32_t)ParticleFlag::secondary, ParticleType::positron);

      // Set this photon to be empty
      photons.cell[tid] = MAX_CELL;
    }
  }
}

}  // namespace Kernels

template <typename PtcClass, typename PhotonClass, typename RadModel>
RadiationTransfer<PtcClass, PhotonClass, RadModel>::RadiationTransfer(
    const Environment& env)
    : m_env(env),
      d_rand_states(nullptr),
      m_threadsPerBlock(512),
      m_blocksPerGrid(512),
      m_numPerBlock(m_blocksPerGrid),
      m_cumNumPerBlock(m_blocksPerGrid),
      m_posInBlock(env.params().max_ptc_number) {
  int seed = m_env.params().random_seed;

  CudaSafeCall(cudaMalloc(
      &d_rand_states,
      m_threadsPerBlock * m_blocksPerGrid * sizeof(curandState)));
  Kernels::init_rand_states<<<m_blocksPerGrid, m_threadsPerBlock>>>(
      (curandState*)d_rand_states, seed);
  CudaCheckError();

  // Allocate auxiliary arrays for pair creation
  // CudaSafeCall(cudaMalloc(&m_numPerBlock, m_blocksPerGrid *
  // sizeof(uint32_t))); CudaSafeCall(cudaMalloc(&m_cumNumPerBlock,
  // m_blocksPerGrid * sizeof(uint32_t)));
  // CudaSafeCall(cudaMalloc(&m_posInBlock,
  // m_env.params().max_ptc_number));
}

template <typename PtcClass, typename PhotonClass, typename RadModel>
RadiationTransfer<PtcClass, PhotonClass,
                  RadModel>::~RadiationTransfer() {
  cudaFree((curandState*)d_rand_states);
}

template <typename PtcClass, typename PhotonClass, typename RadModel>
void
RadiationTransfer<PtcClass, PhotonClass, RadModel>::emit_photons(
    PhotonClass& photons, PtcClass& ptc) {
  m_posInBlock.assign_dev(0, ptc.number());
  m_numPerBlock.assign_dev(0);
  m_cumNumPerBlock.assign_dev(0);

  cudaDeviceSynchronize();
  // Logger::print_debug("Initialize finished");

  Kernels::count_photon_produced<typename PtcClass::DataClass, RadModel>
      <<<m_blocksPerGrid, m_threadsPerBlock>>>(
          ptc.data(), ptc.number(), m_numPerBlock.data_d(),
          m_posInBlock.data_d(), (curandState*)d_rand_states);
  CudaCheckError();

  thrust::device_ptr<int> ptrNumPerBlock(m_numPerBlock.data_d());
  thrust::device_ptr<int> ptrCumNum(m_cumNumPerBlock.data_d());

  cudaDeviceSynchronize();
  // Logger::print_debug("Count finished");
  // Scan the number of photons produced per block. The last one will be
  // the total
  thrust::exclusive_scan(ptrNumPerBlock,
                         ptrNumPerBlock + m_blocksPerGrid, ptrCumNum);
  CudaCheckError();
  // Logger::print_debug("Scan finished");
  m_cumNumPerBlock.sync_to_host();
  m_numPerBlock.sync_to_host();
  int new_photons = m_cumNumPerBlock[m_blocksPerGrid - 1] +
                    m_numPerBlock[m_blocksPerGrid - 1];
  Logger::print_info("{} photons are produced!", new_photons);

  Kernels::produce_photons<typename PtcClass::DataClass,
                           typename PhotonClass::DataClass, RadModel>
      <<<m_blocksPerGrid, m_threadsPerBlock>>>(
          ptc.data(), ptc.number(), photons.data(), photons.number(),
          m_posInBlock.data_d(), m_numPerBlock.data_d(),
          m_cumNumPerBlock.data_d(), (curandState*)d_rand_states);
  CudaCheckError();

  photons.set_num(photons.number() + new_photons);
  Logger::print_info("There are {} photons in the pool",
                     photons.number());
}

template <typename PtcClass, typename PhotonClass, typename RadModel>
void
RadiationTransfer<PtcClass, PhotonClass, RadModel>::produce_pairs(
    PtcClass& ptc, PhotonClass& photons) {
  m_posInBlock.assign_dev(0, ptc.number());
  m_numPerBlock.assign_dev(0);
  m_cumNumPerBlock.assign_dev(0);

  Kernels::count_pairs_produced<typename PhotonClass::DataClass>
      <<<m_blocksPerGrid, m_threadsPerBlock>>>(
          photons.data(), photons.number(), m_numPerBlock.data_d(),
          m_posInBlock.data_d(), (curandState*)d_rand_states);
  CudaCheckError();

  thrust::device_ptr<int> ptrNumPerBlock(m_numPerBlock.data_d());
  thrust::device_ptr<int> ptrCumNum(m_cumNumPerBlock.data_d());

  // Scan the number of photons produced per block. The last one will be
  // the total
  thrust::exclusive_scan(ptrNumPerBlock,
                         ptrNumPerBlock + m_blocksPerGrid, ptrCumNum);
  m_cumNumPerBlock.sync_to_host();
  m_numPerBlock.sync_to_host();
  int new_pairs = (m_cumNumPerBlock[m_blocksPerGrid - 1] +
                   m_numPerBlock[m_blocksPerGrid - 1]);
  Logger::print_info("{} electron-positron pairs are produced!",
                     new_pairs);

  Kernels::produce_pairs<typename PtcClass::DataClass,
                         typename PhotonClass::DataClass>
      <<<m_blocksPerGrid, m_threadsPerBlock>>>(
          photons.data(), photons.number(), ptc.data(), ptc.number(),
          m_posInBlock.data_d(), m_numPerBlock.data_d(),
          m_cumNumPerBlock.data_d(), (curandState*)d_rand_states);
  CudaCheckError();

  ptc.set_num(ptc.number() + new_pairs * 2);
  Logger::print_info("There are {} particles in the pool",
                     ptc.number());
}

////////////////////////////////////////////////////////////////////////
//  Explicit instantiations
////////////////////////////////////////////////////////////////////////

template class RadiationTransfer<Particles_1D, Photons_1D,
                                 InverseComptonDummy<Kernels::CudaRng>>;
template class RadiationTransfer<Particles, Photons,
                                 InverseComptonDummy<Kernels::CudaRng>>;
template class RadiationTransfer<Particles_1D, Photons_1D,
                                 InverseComptonPL1D<Kernels::CudaRng>>;
template class RadiationTransfer<Particles, Photons,
                                 InverseComptonPL1D<Kernels::CudaRng>>;

}  // namespace Aperture