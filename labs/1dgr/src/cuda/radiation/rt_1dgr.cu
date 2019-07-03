#include "core/detail/multi_array_utils.hpp"
#include "cuda/constant_mem.h"
#include "cuda/core/cu_sim_data1d.h"
#include "cuda/core/cu_sim_environment.h"
#include "cuda/cudaUtility.h"
#include "cuda/cuda_control.h"
#include "cuda/cudarng.h"
#include "cuda/data/particles_dev.h"
#include "cuda/data/photons_dev.h"
#include "cuda/grids/grid_1dgr_dev.h"
#include "cuda/kernels.h"
#include "cuda/ptr_util.h"
#include "cuda/radiation/rt_ic_dev.h"
#include "radiation/spectra.h"
#include "rt_1dgr.h"
#include "utils/logger.h"
#include "utils/util_functions.h"
#include <cuda_runtime_api.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

// __device__ void __syncthreads();

namespace Aperture {

namespace Kernels {

template <typename PtcData>
__global__ void count_photon_produced(PtcData ptc, size_t number,
                                      Grid_1dGR_dev::mesh_ptrs mesh_ptrs,
                                      int *ph_count, int *phPos,
                                      curandState *states, Scalar dt) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  __shared__ int photonProduced;
  curandState local_state = states[id];
  if (threadIdx.x == 0)
    photonProduced = 0;

  __syncthreads();

  for (uint32_t tid = id; tid < number; tid += blockDim.x * gridDim.x) {
    uint32_t cell = ptc.cell[tid];
    // Skip empty particles
    if (cell == MAX_CELL)
      continue;
    auto flag = ptc.flag[tid];
    auto x1 = ptc.x1[tid];
    // auto p1 = ptc.p1[tid];
    int sp = get_ptc_type(flag);

    Scalar alpha =
        mesh_ptrs.alpha[cell] * x1 + mesh_ptrs.alpha[cell - 1] * (1.0f - x1);
    // Skip photon emission when outside given radius
    Scalar gamma = alpha * ptc.E[tid];

    float u = curand_uniform(&local_state);
    // TODO: Add a scaling factor here for the rate that may depend on
    // position
    if (u < find_ic_rate(gamma) * alpha * dt) {
      phPos[tid] = atomicAdd(&photonProduced, 1) + 1;
    }
  }

  __syncthreads();

  // Record the number of photons produced this block to global array
  if (threadIdx.x == 0) {
    ph_count[blockIdx.x] = photonProduced;
  }
  states[id] = local_state;
}

template <typename PtcData, typename PhotonData>
__global__ void
produce_photons(PtcData ptc, size_t ptc_num, PhotonData photons, size_t ph_num,
                Grid_1dGR_dev::mesh_ptrs mesh_ptrs, int *phPos, int *ph_count,
                int *ph_cum, curandState *states) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  curandState local_state = states[id];

  for (uint32_t tid = id; tid < ptc_num; tid += blockDim.x * gridDim.x) {
    int pos_in_block = phPos[tid] - 1;
    uint32_t cell = ptc.cell[tid];
    if (pos_in_block > -1 && cell != MAX_CELL) {
      int start_pos = ph_cum[blockIdx.x];

      Scalar x1 = ptc.x1[tid];
      Scalar p1 = ptc.p1[tid];
      Scalar u0_ptc = ptc.E[tid];
      uint32_t c = ptc.cell[tid];
      Scalar xi = dev_mesh.pos(0, c, x1);
      // FIXME: pass a in as a parameter
      Scalar a = dev_params.a;
      const Scalar rp = 1.0f + std::sqrt(1.0f - a * a);
      const Scalar rm = 1.0f - std::sqrt(1.0f - a * a);
      Scalar exp_xi = std::exp(xi * (rp - rm));
      Scalar r = (rp - rm * exp_xi) / (1.0 - exp_xi);
      Scalar Delta = r * r - 2.0 * r + a * a;

      Scalar alpha =
          mesh_ptrs.alpha[cell] * x1 + mesh_ptrs.alpha[cell - 1] * (1.0f - x1);
      Scalar D1 = mesh_ptrs.D1[c] * x1 + mesh_ptrs.D1[c - 1] * (1.0f - x1);
      Scalar D2 = mesh_ptrs.D2[c] * x1 + mesh_ptrs.D2[c - 1] * (1.0f - x1);
      Scalar D3 = mesh_ptrs.D3[c] * x1 + mesh_ptrs.D3[c - 1] * (1.0f - x1);
      Scalar B3B1 =
          mesh_ptrs.B3B1[c] * x1 + mesh_ptrs.B3B1[c - 1] * (1.0f - x1);
      Scalar g11 =
          mesh_ptrs.gamma_rr[c] * x1 + mesh_ptrs.gamma_rr[c - 1] * (1.0f - x1);
      Scalar g33 =
          mesh_ptrs.gamma_ff[c] * x1 + mesh_ptrs.gamma_ff[c - 1] * (1.0f - x1);
      Scalar beta =
          mesh_ptrs.beta_phi[c] * x1 + mesh_ptrs.beta_phi[c - 1] * (1.0f - x1);

      Scalar ur_ptc = u0_ptc * Delta * (p1 / u0_ptc - D1) / D2;
      // Scalar uphi_ptc = dev_params.omega * u0_ptc + B3B1 * ur_ptc;

      Scalar gamma = alpha * ptc.E[tid];
      Scalar Eph = gen_photon_e(gamma, &local_state);
      // Limit energy loss so that remaining particle momentum still
      // makes sense
      if (Eph >= gamma - 1.01f)
        Eph = gamma - 1.01f;

      ptc.E[tid] = (gamma - Eph) / alpha;
      ptc.p1[tid] =
          sgn(p1) *
          std::sqrt(square(ptc.E[tid]) * (D2 * (alpha * alpha - D3) + D1 * D1) -
                    D2);
      // if p1 becomes NaN, set it to zero
      if (ptc.p1[tid] != ptc.p1[tid])
        ptc.p1[tid] = 0.0f;

      // If photon energy is too low, do not track it, but still
      // subtract its energy as done above
      if (std::abs(Eph) < 0.01f / dev_params.e_min)
        continue;

      // Add the new photon
      // Scalar path = rad_model.draw_photon_freepath(Eph);
      // printf("Eph is %f, path is %f\n", Eph, path);
      int offset = ph_num + start_pos + pos_in_block;
      photons.x1[offset] = ptc.x1[tid];
      photons.p1[offset] = (Eph / gamma) * ur_ptc / g11;
      photons.pf[offset] =
          (Eph / gamma) * (u0_ptc * (dev_params.omega + beta) + B3B1 * ur_ptc) /
          g33;
      // photons.E[offset] = sgn(photons.p1[offset]) * Eph / alpha;
      photons.E[offset] = Eph / alpha;
      photons.weight[offset] = ptc.weight[tid];
      photons.cell[offset] = ptc.cell[tid];
      float u = curand_uniform(&local_state);
      photons.flag[offset] =
          (u < dev_params.track_percent ? bit_or(PhotonFlag::tracked) : 0);
      if (id == 0) {
        printf("p1 %f, pf %f, E %f\n", photons.p1[offset], photons.pf[offset], photons.E[offset]);
      }
    }
  }
  states[id] = local_state;
}

template <typename PhotonData>
__global__ void count_pairs_produced(PhotonData photons, size_t number,
                                     Grid_1dGR_dev::mesh_ptrs mesh_ptrs,
                                     int *pair_count, int *pair_pos,
                                     curandState *states, Scalar dt) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  CudaRng rng(&states[id]);
  __shared__ int pairsProduced;
  if (threadIdx.x == 0)
    pairsProduced = 0;

  __syncthreads();

  for (uint32_t tid = id; tid < number; tid += blockDim.x * gridDim.x) {
    // if (tid >= number) continue;
    uint32_t cell = photons.cell[tid];
    if (cell == MAX_CELL)
      continue;
    if (!dev_mesh.is_in_bulk(cell)) {
      photons.cell[tid] = MAX_CELL;
      continue;
    }

    auto x1 = photons.x1[tid];

    Scalar alpha =
        mesh_ptrs.alpha[cell] * x1 + mesh_ptrs.alpha[cell - 1] * (1.0f - x1);
    // Skip photon emission when outside given radius
    Scalar u0_hat = alpha * std::abs(photons.E[tid]);
    if (u0_hat < dev_params.E_ph_min) {
      photons.cell[tid] = MAX_CELL;
      continue;
    }

    Scalar prob = find_gg_rate(u0_hat) * alpha * dt;
    if (tid == 0)
      printf("u0_hat is %f, prob is %f\n", u0_hat, prob);

    float u = rng();
    if (u < prob) {
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
produce_pairs(PhotonData photons, size_t ph_num, PtcData ptc, size_t ptc_num,
              Grid_1dGR_dev::mesh_ptrs mesh_ptrs, int *pair_pos,
              int *pair_count, int *pair_cum, curandState *states) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  CudaRng rng(&states[id]);
  for (uint32_t tid = id; tid < ph_num; tid += blockDim.x * gridDim.x) {
    int pos_in_block = pair_pos[tid] - 1;
    if (pos_in_block > -1 && photons.cell[tid] != MAX_CELL) {
      int start_pos = pair_cum[blockIdx.x] * 2;

      // Split the photon energy evenly between the pairs
      Scalar u0 = 0.5f * std::abs(photons.E[tid]);

      uint32_t c = photons.cell[tid];
      Pos_t x1 = photons.x1[tid];
      // Set this photon to be empty
      photons.cell[tid] = MAX_CELL;

      // Scalar xi = dev_mesh.pos(0, c, x1);
      // // FIXME: pass a in as a parameter
      // constexpr Scalar a = 0.99;
      // const Scalar rp = 1.0f + std::sqrt(1.0f - a * a);
      // const Scalar rm = 1.0f - std::sqrt(1.0f - a * a);
      // Scalar exp_xi = std::exp(xi * (rp - rm));
      // Scalar r = (rp - rm * exp_xi) / (1.0 - exp_xi);
      // Scalar Delta = r * r - 2.0 * r + a * a;

      Scalar alpha =
          mesh_ptrs.alpha[c] * x1 + mesh_ptrs.alpha[c - 1] * (1.0f - x1);
      Scalar D1 = mesh_ptrs.D1[c] * x1 + mesh_ptrs.D1[c - 1] * (1.0f - x1);
      Scalar D2 = mesh_ptrs.D2[c] * x1 + mesh_ptrs.D2[c - 1] * (1.0f - x1);
      Scalar D3 = mesh_ptrs.D3[c] * x1 + mesh_ptrs.D3[c - 1] * (1.0f - x1);

      Scalar p1 =
          sgn(photons.p1[tid]) *
          std::sqrt(u0 * u0 * (D2 * (alpha * alpha - D3) + D1 * D1) - D2);
      if (p1 != p1)
        p1 = 0.0f;

      // Add the two new particles
      int offset_e = ptc_num + start_pos + pos_in_block * 2;
      int offset_p = ptc_num + start_pos + pos_in_block * 2 + 1;

      ptc.x1[offset_e] = ptc.x1[offset_p] = x1;

      ptc.p1[offset_e] = ptc.p1[offset_p] = p1;
      ptc.E[offset_e] = ptc.E[offset_p] = u0;

      if (tid < 10000) {
        printf("pair u0 %f, p1 %f\n", u0, p1);
      }

#ifndef NDEBUG
      assert(ptc.cell[offset_e] == MAX_CELL);
      assert(ptc.cell[offset_p] == MAX_CELL);
#endif
      ptc.weight[offset_e] = ptc.weight[offset_p] = photons.weight[tid];
      ptc.cell[offset_e] = ptc.cell[offset_p] = c;
      float u = rng();
      ptc.flag[offset_e] = set_ptc_type_flag(
          (u < dev_params.track_percent
               ? bit_or(ParticleFlag::secondary, ParticleFlag::tracked)
               : bit_or(ParticleFlag::secondary)),
          ParticleType::electron);
      ptc.flag[offset_p] = set_ptc_type_flag(
          (u < dev_params.track_percent
               ? bit_or(ParticleFlag::secondary, ParticleFlag::tracked)
               : bit_or(ParticleFlag::secondary)),
          ParticleType::positron);

    }
  }
}

} // namespace Kernels

RadiationTransfer1DGR::RadiationTransfer1DGR(const cu_sim_environment &env)
    : m_env(env), d_rand_states(nullptr), m_threadsPerBlock(256),
      m_blocksPerGrid(512), m_numPerBlock(m_blocksPerGrid),
      m_cumNumPerBlock(m_blocksPerGrid),
      m_posInBlock(env.params().max_ptc_number), m_ic(env.params()) {
  int seed = m_env.params().random_seed;

  CudaSafeCall(cudaMalloc(&d_rand_states, m_threadsPerBlock * m_blocksPerGrid *
                                              sizeof(curandState)));
  init_rand_states((curandState *)d_rand_states, seed, m_threadsPerBlock,
                   m_blocksPerGrid);

  // Init inverse compton module
  // Spectra::black_body ne(env.params().star_kT);
  Spectra::broken_power_law ne(1.25, 1.1, m_env.params().e_min, 1.0e-10, 0.1);
  // FIXME: Insert correct normalization here for the background photon
  // field
  // m_ic.init(ne, ne.emin(), ne.emax(), 1.23529e18 / m_env.params().ic_path);
  m_ic.init(ne, ne.emin(), ne.emax(), 1.50e24 / m_env.params().ic_path);
}

RadiationTransfer1DGR::~RadiationTransfer1DGR() {
  CudaSafeCall(cudaFree((curandState *)d_rand_states));
}

void RadiationTransfer1DGR::emit_photons(cu_sim_data1d &data, Scalar dt) {
  auto &ptc = data.particles;
  auto &photons = data.photons;
  m_posInBlock.assign_dev(0, ptc.number());
  m_numPerBlock.assign_dev(0);
  m_cumNumPerBlock.assign_dev(0);

  cudaDeviceSynchronize();
  const Grid_1dGR_dev &grid =
      *dynamic_cast<const Grid_1dGR_dev *>(data.grid.get());
  auto mesh_ptrs = grid.get_mesh_ptrs();

  // Count the number of photons produced
  Kernels::count_photon_produced<particle1d_data>
      <<<m_blocksPerGrid, m_threadsPerBlock>>>(
          ptc.data(), ptc.number(), mesh_ptrs, m_numPerBlock.data_d(),
          m_posInBlock.data_d(), (curandState *)d_rand_states, dt);
  CudaCheckError();

  thrust::device_ptr<int> ptrNumPerBlock(m_numPerBlock.data_d());
  thrust::device_ptr<int> ptrCumNum(m_cumNumPerBlock.data_d());
  // Scan the number of photons produced per block. The result gives the
  // offset for each block
  thrust::exclusive_scan(ptrNumPerBlock, ptrNumPerBlock + m_blocksPerGrid,
                         ptrCumNum);
  CudaCheckError();
  m_cumNumPerBlock.sync_to_host();
  m_numPerBlock.sync_to_host();
  int new_photons = m_cumNumPerBlock[m_blocksPerGrid - 1] +
                    m_numPerBlock[m_blocksPerGrid - 1];
  Logger::print_info("{} photons are produced!", new_photons);

  // Actually produce the photons
  Kernels::produce_photons<particle1d_data, photon1d_data>
      <<<m_blocksPerGrid, m_threadsPerBlock>>>(
          ptc.data(), ptc.number(), photons.data(), photons.number(), mesh_ptrs,
          m_posInBlock.data_d(), m_numPerBlock.data_d(),
          m_cumNumPerBlock.data_d(), (curandState *)d_rand_states);
  CudaCheckError();

  int padding = 1;
  photons.set_num(photons.number() + new_photons + padding);
}

void RadiationTransfer1DGR::produce_pairs(cu_sim_data1d &data, Scalar dt) {
  auto &ptc = data.particles;
  auto &photons = data.photons;
  thrust::device_ptr<int> ptrNumPerBlock(m_numPerBlock.data_d());
  thrust::device_ptr<int> ptrCumNum(m_cumNumPerBlock.data_d());
  int new_pairs;

  m_posInBlock.assign_dev(0, photons.number());
  m_numPerBlock.assign_dev(0);
  m_cumNumPerBlock.assign_dev(0);

  cudaDeviceSynchronize();
  const Grid_1dGR_dev &grid =
      *dynamic_cast<const Grid_1dGR_dev *>(data.grid.get());
  auto mesh_ptrs = grid.get_mesh_ptrs();

  // Count the number of photons produced
  Kernels::count_pairs_produced<photon1d_data>
      <<<m_blocksPerGrid, m_threadsPerBlock>>>(
          photons.data(), photons.number(), mesh_ptrs, m_numPerBlock.data_d(),
          m_posInBlock.data_d(), (curandState *)d_rand_states, dt);
  CudaCheckError();

  // Scan the number of photons produced per block. The result gives the
  // offset for each block
  thrust::exclusive_scan(ptrNumPerBlock, ptrNumPerBlock + m_blocksPerGrid,
                         ptrCumNum);
  CudaCheckError();
  m_cumNumPerBlock.sync_to_host();
  m_numPerBlock.sync_to_host();
  new_pairs = m_cumNumPerBlock[m_blocksPerGrid - 1] +
              m_numPerBlock[m_blocksPerGrid - 1];
  Logger::print_info("{} electron-positron pairs are produced!", new_pairs);

  // Actually produce the photons
  Kernels::produce_pairs<particle1d_data, photon1d_data>
      <<<m_blocksPerGrid, m_threadsPerBlock>>>(
          photons.data(), photons.number(), ptc.data(), ptc.number(), mesh_ptrs,
          m_posInBlock.data_d(), m_numPerBlock.data_d(),
          m_cumNumPerBlock.data_d(), (curandState *)d_rand_states);
  CudaCheckError();

  int padding = 10;
  ptc.set_num(ptc.number() + new_pairs * 2 + padding);

  CudaSafeCall(cudaDeviceSynchronize());
}

} // namespace Aperture
