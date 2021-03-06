#include "core/detail/multi_array_utils.hpp"
#include "cuda/constant_mem.h"
#include "cuda/cudaUtility.h"
#include "cuda/cuda_control.h"
#include "cuda/cudarng.h"
#include "cuda/data/particles_dev.h"
#include "cuda/kernels.h"
#include "cuda/ptr_util.h"
// #include "data/particles_1d.h"
#include "cuda/data/photons_dev.h"
// #include "data/photons_1d.h"
// #include "radiation/curvature_instant.h"
#include "cuda/core/cu_sim_data.h"
#include "cuda/core/cu_sim_environment.h"
#include "cuda/radiation/rt_magnetar.h"
#include "utils/logger.h"
#include "utils/util_functions.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

namespace Aperture {

namespace Kernels {

template <typename PtcData>
__global__ void
count_photon_produced(PtcData ptc, size_t number, int* ph_count,
                      int* phPos, curandState* states,
                      cudaPitchedPtr ph_events) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;

  __shared__ int photonProduced;
  if (threadIdx.x == 0) photonProduced = 0;

  __syncthreads();

  for (uint32_t tid = id; tid < number; tid += blockDim.x * gridDim.x) {
    uint32_t cell = ptc.cell[tid];
    // Skip empty particles
    if (cell == MAX_CELL) continue;
    auto flag = ptc.flag[tid];
    int sp = get_ptc_type(flag);
    if (sp == (int)ParticleType::ion ||
        !check_bit(flag, ParticleFlag::emit_photon))
      continue;
    int c1 = dev_mesh.get_c1(cell);
    int c2 = dev_mesh.get_c2(cell);

    // Skip photon emission when outside given radius
    Scalar w = ptc.weight[tid];

    // if (rad_model.emit_photon(gamma)) {
    // if (gamma > dev_params.gamma_thr && r < dev_params.r_cutoff &&
    //     r > 1.02f) {
    // phPos[tid] = atomicAdd_block(&photonProduced, 1) + 1;
    phPos[tid] = atomicAdd(&photonProduced, 1) + 1;
    atomicAdd(
        ptrAddr(ph_events, c2 * ph_events.pitch + c1 * sizeof(Scalar)),
        w);
    ptc.flag[tid] &= ~bit_or(ParticleFlag::emit_photon);
    // }
  }

  __syncthreads();

  // Record the number of photons produced this block to global array
  if (threadIdx.x == 0) {
    ph_count[blockIdx.x] = photonProduced;
  }
}

template <typename PtcData, typename PhotonData>
__global__ void
produce_photons(PtcData ptc, size_t ptc_num, PhotonData photons,
                size_t ph_num, int* phPos, int* ph_count, int* ph_cum,
                cudaPitchedPtr b1, cudaPitchedPtr b2, cudaPitchedPtr b3,
                curandState* states) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  CudaRng rng(&states[id]);
  for (uint32_t tid = id; tid < ptc_num;
       tid += blockDim.x * gridDim.x) {
    int pos_in_block = phPos[tid] - 1;
    if (pos_in_block > -1 && ptc.cell[tid] != MAX_CELL) {
      int start_pos = ph_cum[blockIdx.x];
      auto c = ptc.cell[tid];
      int c1 = dev_mesh.get_c1(c);
      int c2 = dev_mesh.get_c2(c);
      Scalar u = rng();

      Scalar B1 = *ptrAddr(b1, c1, c2);
      Scalar B2 = *ptrAddr(b2, c1, c2);
      Scalar B3 = *ptrAddr(b3, c1, c2);
      Scalar b = std::sqrt(B1 * B1 + B2 * B2 + B3 * B3) / dev_params.BQ;

      Scalar p1 = ptc.p1[tid];
      Scalar p2 = ptc.p2[tid];
      Scalar p3 = ptc.p3[tid];
      Scalar gamma = ptc.E[tid];

      Scalar pi = std::sqrt(gamma * gamma - 1.0f);
      Scalar Eph = (gamma - pi * (2.0f * u - 1.0f)) *
                   (1.0f - 1.0f / std::sqrt(1.0f + 2.0f * b));
      if (Eph > gamma - 1.0f) Eph = gamma - 1.1f;

      Scalar pf = std::sqrt(square(gamma - Eph) - 1.0f);
      ptc.p1[tid] = p1 * pf / pi;
      ptc.p2[tid] = p2 * pf / pi;
      ptc.p3[tid] = p3 * pf / pi;
      ptc.E[tid] = gamma - Eph;

      Scalar path = dev_params.photon_path;
      // If photon energy is too low, do not track it, but still
      // subtract its energy as done above
      if (Eph < dev_params.E_ph_min) continue;
      if (c2 < dev_mesh.guard[1] + 20)
          // c2 >= dev_mesh.dims[1] - dev_mesh.guard[1] - 20)
        continue;

      // Add the new photon
      int offset = ph_num + start_pos + pos_in_block;
      photons.x1[offset] = ptc.x1[tid];
      photons.x2[offset] = ptc.x2[tid];
      photons.x3[offset] = ptc.x3[tid];
      photons.p1[offset] = Eph * p1 / pi;
      photons.p2[offset] = Eph * p2 / pi;
      photons.p3[offset] = Eph * p3 / pi;
      photons.weight[offset] = ptc.weight[tid];
      photons.path_left[offset] = path;
      photons.cell[offset] = ptc.cell[tid];
    }
  }
}

template <typename PhotonData>
__global__ void
count_pairs_produced(PhotonData photons, size_t number, int* pair_count,
                     int* pair_pos, curandState* states,
                     cudaPitchedPtr pair_events, cudaPitchedPtr rho0,
                     cudaPitchedPtr rho1, cudaPitchedPtr rho2) {
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
    int c1 = dev_mesh.get_c1(cell);
    int c2 = dev_mesh.get_c2(cell);
    // Remove photon if it is too close to the axis
    if (c2 <= dev_mesh.guard[1] + 4 ||
        c2 >= dev_mesh.dims[1] - dev_mesh.guard[1] - 5) {
      photons.cell[tid] = MAX_CELL;
      continue;
    }
    if (!dev_mesh.is_in_bulk(c1, c2)) {
      photons.cell[tid] = MAX_CELL;
      continue;
    }

    // Get the B field at the current location
    // Scalar B1v = interp(b1, photons.x1[tid], photons.x2[tid], c1, c2,
    // Stagger(0b001)); Scalar B2v = interp(b2, photons.x1[tid],
    // photons.x2[tid], c1, c2, Stagger(0b010)); Scalar B3v = interp(b3,
    // photons.x1[tid], photons.x2[tid], c1, c2, Stagger(0b100));

    if (photons.path_left[tid] <= 0.0f) {
      // if (*ptrAddr(rho0, c1, c2))
      Scalar rho =
          max(std::abs(*ptrAddr(rho1, c1, c2) + *ptrAddr(rho0, c1, c2)),
              0.0001f);
      Scalar N = max(*ptrAddr(rho1, c1, c2), -*ptrAddr(rho0, c1, c2));
      // Scalar multiplicity = N / rho;
      if (N > 1.0e6f) {
        photons.cell[tid] = MAX_CELL;
        continue;
      }
      pair_pos[tid] = atomicAdd(&pairsProduced, 1) + 1;
      int c1 = dev_mesh.get_c1(cell);
      Scalar w = photons.weight[tid];

      atomicAdd(ptrAddr(pair_events,
                        c2 * pair_events.pitch + c1 * sizeof(Scalar)),
                w);
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
  CudaRng rng(&states[id]);
  // auto inv_comp = make_inverse_compton_PL1D(dev_params,
  // dev_params.photon_path, rng); RadModel rad_model(dev_params, rng);

  for (uint32_t tid = id; tid < ph_num; tid += blockDim.x * gridDim.x) {
    int pos_in_block = pair_pos[tid] - 1;
    if (pos_in_block > -1 && photons.cell[tid] != MAX_CELL) {
      int start_pos = pair_cum[blockIdx.x] * 2;

      // Split the photon energy evenly between the pairs
      Scalar p1 = photons.p1[tid];
      Scalar p2 = photons.p2[tid];
      Scalar p3 = photons.p3[tid];
      Scalar E_ph2 = p1 * p1 + p2 * p2 + p3 * p3;
      // Scalar new_p = std::sqrt(max(0.25f * E_ph * E_ph, 1.0f)
      // - 1.0f);
      Scalar ratio = std::sqrt(0.25f - 1.0f / E_ph2);

      // Add the two new particles
      int offset_e = ptc_num + start_pos + pos_in_block * 2;
      int offset_p = ptc_num + start_pos + pos_in_block * 2 + 1;
      // int offset_p = ptc_num + start_pos + pos_in_block +
      // pair_count[blockIdx.x];
      float u = rng();

      ptc.x1[offset_e] = ptc.x1[offset_p] = photons.x1[tid];
      ptc.x2[offset_e] = ptc.x2[offset_p] = photons.x2[tid];
      ptc.x3[offset_e] = ptc.x3[offset_p] = photons.x3[tid];
      // printf("x1 = %f, x2 = %f, x3 = %f\n", ptc.x1[offset_e],
      // ptc.x2[offset_e], ptc.x3[offset_e]);

      ptc.p1[offset_e] = ptc.p1[offset_p] = ratio * p1;
      ptc.p2[offset_e] = ptc.p2[offset_p] = ratio * p2;
      ptc.p3[offset_e] = ptc.p3[offset_p] = ratio * p3;
      ptc.E[offset_e] = ptc.E[offset_p] =
          sqrt(1.0f + ratio * ratio * E_ph2);

#ifndef NDEBUG
      assert(ptc.cell[offset_e] == MAX_CELL);
      assert(ptc.cell[offset_p] == MAX_CELL);
#endif
      ptc.weight[offset_e] = ptc.weight[offset_p] = photons.weight[tid];
      ptc.cell[offset_e] = ptc.cell[offset_p] = photons.cell[tid];
      ptc.flag[offset_e] =
          set_ptc_type_flag(bit_or(ParticleFlag::secondary,
                                   (u < dev_params.track_percent
                                        ? ParticleFlag::tracked
                                        : ParticleFlag::nothing)),
                            ParticleType::electron);
      ptc.flag[offset_p] =
          set_ptc_type_flag(bit_or(ParticleFlag::secondary,
                                   (u < dev_params.track_percent
                                        ? ParticleFlag::tracked
                                        : ParticleFlag::nothing)),
                            ParticleType::positron);

      // Set this photon to be empty
      photons.cell[tid] = MAX_CELL;
    }
  }
}

}  // namespace Kernels

RadiationTransferMagnetar::RadiationTransferMagnetar(
    const cu_sim_environment& env)
    : m_env(env),
      d_rand_states(nullptr),
      m_threadsPerBlock(256),
      m_blocksPerGrid(512),
      m_numPerBlock(m_blocksPerGrid),
      m_cumNumPerBlock(m_blocksPerGrid),
      m_posInBlock(env.params().max_ptc_number),
      m_pair_events(env.local_grid()),
      m_ph_events(env.local_grid()) {
  int seed = m_env.params().random_seed;

  CudaSafeCall(cudaMalloc(
      &d_rand_states,
      m_threadsPerBlock * m_blocksPerGrid * sizeof(curandState)));
  init_rand_states((curandState*)d_rand_states, seed, m_threadsPerBlock,
                   m_blocksPerGrid);

  m_pair_events.initialize();
  m_ph_events.initialize();
  // Kernels::init_rand_states<<<m_blocksPerGrid, m_threadsPerBlock>>>(
  //     (curandState*)d_rand_states, seed);
  // CudaCheckError();

  // Allocate auxiliary arrays for pair creation
  // CudaSafeCall(cudaMalloc(&m_numPerBlock, m_blocksPerGrid *
  // sizeof(uint32_t))); CudaSafeCall(cudaMalloc(&m_cumNumPerBlock,
  // m_blocksPerGrid * sizeof(uint32_t)));
  // CudaSafeCall(cudaMalloc(&m_posInBlock,
  // m_env.params().max_ptc_number));
}

RadiationTransferMagnetar::~RadiationTransferMagnetar() {
  cudaFree((curandState*)d_rand_states);
}

void
RadiationTransferMagnetar::emit_photons(cu_sim_data& data) {
  auto& ptc = data.particles;
  auto& photons = data.photons;
  m_posInBlock.assign_dev(0, ptc.number());
  m_numPerBlock.assign_dev(0);
  m_cumNumPerBlock.assign_dev(0);

  cudaDeviceSynchronize();
  // Logger::print_debug("Initialize finished");

  Kernels::count_photon_produced<particle_data>
      <<<m_blocksPerGrid, m_threadsPerBlock>>>(
          ptc.data(), ptc.number(), m_numPerBlock.data_d(),
          m_posInBlock.data_d(), (curandState*)d_rand_states,
          m_ph_events.ptr());
  CudaCheckError();

  thrust::device_ptr<int> ptrNumPerBlock(m_numPerBlock.data_d());
  thrust::device_ptr<int> ptrCumNum(m_cumNumPerBlock.data_d());

  // cudaDeviceSynchronize();
  // Logger::print_debug("Count finished");
  // Scan the number of photons produced per block. The result gives the
  // offset for each block
  thrust::exclusive_scan(ptrNumPerBlock,
                         ptrNumPerBlock + m_blocksPerGrid, ptrCumNum);
  CudaCheckError();
  // Logger::print_debug("Scan finished");
  m_cumNumPerBlock.copy_to_host();
  m_numPerBlock.copy_to_host();
  int new_photons = m_cumNumPerBlock[m_blocksPerGrid - 1] +
                    m_numPerBlock[m_blocksPerGrid - 1];
  Logger::print_info("{} photons are produced!", new_photons);

  Kernels::produce_photons<particle_data, photon_data>
      <<<m_blocksPerGrid, m_threadsPerBlock>>>(
          ptc.data(), ptc.number(), photons.data(), photons.number(),
          m_posInBlock.data_d(), m_numPerBlock.data_d(),
          m_cumNumPerBlock.data_d(), data.B.ptr(0), data.B.ptr(1),
          data.B.ptr(2), (curandState*)d_rand_states);
  CudaCheckError();

  int padding = 1;
  photons.set_num(photons.number() + new_photons + padding);
  // Logger::print_info("There are {} photons in the pool",
  //                    photons.number());
  // cudaDeviceSynchronize();
}

void
RadiationTransferMagnetar::produce_pairs(cu_sim_data& data) {
  auto& ptc = data.particles;
  auto& photons = data.photons;
  m_posInBlock.assign_dev(0, photons.number());
  m_numPerBlock.assign_dev(0);
  m_cumNumPerBlock.assign_dev(0);

  Kernels::count_pairs_produced<photon_data>
      <<<m_blocksPerGrid, m_threadsPerBlock>>>(
          photons.data(), photons.number(), m_numPerBlock.data_d(),
          m_posInBlock.data_d(), (curandState*)d_rand_states,
          m_pair_events.ptr(), data.Rho[0].ptr(), data.Rho[1].ptr(),
          data.Rho[2].ptr());
  CudaCheckError();

  thrust::device_ptr<int> ptrNumPerBlock(m_numPerBlock.data_d());
  thrust::device_ptr<int> ptrCumNum(m_cumNumPerBlock.data_d());
  // cudaDeviceSynchronize();

  // Scan the number of photons produced per block. The last one will be
  // the total
  thrust::exclusive_scan(ptrNumPerBlock,
                         ptrNumPerBlock + m_blocksPerGrid, ptrCumNum);
  m_cumNumPerBlock.copy_to_host();
  m_numPerBlock.copy_to_host();
  int new_pairs = (m_cumNumPerBlock[m_blocksPerGrid - 1] +
                   m_numPerBlock[m_blocksPerGrid - 1]);
  Logger::print_info("{} electron-positron pairs are produced!",
                     new_pairs);

  Kernels::produce_pairs<particle_data, photon_data>
      <<<m_blocksPerGrid, m_threadsPerBlock>>>(
          photons.data(), photons.number(), ptc.data(), ptc.number(),
          m_posInBlock.data_d(), m_numPerBlock.data_d(),
          m_cumNumPerBlock.data_d(), (curandState*)d_rand_states);
  CudaCheckError();

  int padding = 1;
  ptc.set_num(ptc.number() + new_pairs * 2 + padding);
  // Logger::print_info("There are {} particles in the pool",
  //                    ptc.number());
  cudaDeviceSynchronize();
}

}  // namespace Aperture