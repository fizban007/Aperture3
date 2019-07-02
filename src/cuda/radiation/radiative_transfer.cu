#include "core/detail/multi_array_utils.hpp"
#include "cuda/constant_mem.h"
#include "cuda/cudaUtility.h"
#include "cuda/cuda_control.h"
#include "cuda/cudarng.h"
#include "cuda/data_ptrs.h"
#include "cuda/kernels.h"
#include "cuda/ptr_util.h"
#include "cuda/utils/iterate_devices.h"
#include "cuda/utils/pitchptr.cuh"
#include "radiation/radiative_transfer.h"
#include "sim_data.h"
#include "sim_environment.h"
#include "utils/logger.h"
#include "utils/timer.h"
#include "utils/util_functions.h"
#include <cuda.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include "cuda/radiation/user_radiative_functions.cuh"

namespace Aperture {

namespace Kernels {

template <typename PtcData>
__global__ void
count_photon_produced(data_ptrs data, size_t number, int *ph_count,
                      int *phPos, curandState *states) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  CudaRng rng(&states[id]);

  __shared__ int photonProduced;
  if (threadIdx.x == 0) photonProduced = 0;

  __syncthreads();

  auto &ptc = data.particles;

  for (uint32_t tid = id; tid < number; tid += blockDim.x * gridDim.x) {
    uint32_t cell = ptc.cell[tid];
    // Skip empty particles
    if (cell == MAX_CELL) continue;
    if (!dev_mesh.is_in_bulk(cell)) continue;
    auto flag = ptc.flag[tid];
    int sp = get_ptc_type(flag);
    if (sp == (int)ParticleType::ion) continue;

    if (check_emit_photon(data, tid, rng)) {
      int c1 = dev_mesh.get_c1(cell);
      int c2 = dev_mesh.get_c2(cell);
      int c3 = dev_mesh.get_c3(cell);
      Scalar w = ptc.weight[tid];

      phPos[tid] = atomicAdd(&photonProduced, 1) + 1;
      atomicAdd(&data.photon_produced(c1, c2, c3), w);
    }
  }

  __syncthreads();

  // Record the number of photons produced this block to global array
  if (threadIdx.x == 0) {
    ph_count[blockIdx.x] = photonProduced;
  }
}

template <typename PtcData, typename PhotonData>
__global__ void
produce_photons(data_ptrs data, size_t ptc_num, size_t ph_num,
                int *phPos, int *ph_count, int *ph_cum,
                curandState *states) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  CudaRng rng(&states[id]);

  for (uint32_t tid = id; tid < ptc_num;
       tid += blockDim.x * gridDim.x) {
    int pos_in_block = phPos[tid] - 1;
    uint32_t cell = data.particles.cell[tid];
    if (pos_in_block > -1 && cell != MAX_CELL) {
      if (!dev_mesh.is_in_bulk(cell)) continue;
      int start_pos = ph_cum[blockIdx.x];

      int offset = ph_num + start_pos + pos_in_block;

      emit_photon(data, tid, offset, rng);
    }
  }
}

template <typename PhotonData>
__global__ void
count_pairs_produced(data_ptrs data, size_t number, int *pair_count,
                     int *pair_pos, curandState *states) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  CudaRng rng(&states[id]);

  __shared__ int pairsProduced;
  if (threadIdx.x == 0) pairsProduced = 0;

  __syncthreads();

  for (uint32_t tid = id; tid < number; tid += blockDim.x * gridDim.x) {
    uint32_t cell = data.photons.cell[tid];
    // Skip empty photons
    if (cell == MAX_CELL || !dev_mesh.is_in_bulk(cell)) continue;
    //   // Remove photon if it is too close to the axis
    //   Scalar theta = dev_mesh.pos(1, c2, photons.x2[tid]);
    //   if (theta < dev_mesh.delta[1] ||
    //       theta > CONST_PI - dev_mesh.delta[1]) {
    //     photons.cell[tid] = MAX_CELL;
    //     continue;
    //   }

    if (check_produce_pair(data, tid, rng)) {
      //     Scalar rho = max(std::abs(rho1(c1, c2) + rho0(c1, c2)),
      //     0.0001f); Scalar N = rho1(c1, c2) - rho0(c1, c2); Scalar
      //     multiplicity = N / rho;
      //     // if (multiplicity > 20.0f) {
      //     //   photons.cell[tid] = MAX_CELL;
      //     //   continue;
      //     // }
      pair_pos[tid] = atomicAdd(&pairsProduced, 1) + 1;
      int c1 = dev_mesh.get_c1(cell);
      int c2 = dev_mesh.get_c2(cell);
      int c3 = dev_mesh.get_c3(cell);
      Scalar w = data.photons.weight[tid];

      atomicAdd(&data.pair_produced(c1, c2, c3), w);
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
produce_pairs(data_ptrs data, size_t ph_num, size_t ptc_num,
              int *pair_pos, int *pair_count, int *pair_cum,
              curandState *states) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  CudaRng rng(&states[id]);

  for (uint32_t tid = id; tid < ph_num; tid += blockDim.x * gridDim.x) {
    int pos_in_block = pair_pos[tid] - 1;
    uint32_t cell = data.photons.cell[tid];
    if (pos_in_block > -1 && cell != MAX_CELL) {
      if (!dev_mesh.is_in_bulk(cell)) continue;
      int start_pos = pair_cum[blockIdx.x] * 2;

      uint32_t offset = ptc_num + start_pos + pos_in_block * 2;

      produce_pair(data, tid, offset, rng);
    }
  }
}

}  // namespace Kernels

radiative_transfer::radiative_transfer(sim_environment &env)
    : m_env(env), m_threadsPerBlock(256), m_blocksPerGrid(512) {
  m_numPerBlock = array<int>(m_blocksPerGrid);
  m_cumNumPerBlock = array<int>(m_blocksPerGrid);
  m_posInBlock = array<int>(m_env.params().max_ptc_number);
  initialize();
}

radiative_transfer::~radiative_transfer() {}

void
radiative_transfer::initialize() {
  user_rt_init(m_env);
}

void
radiative_transfer::emit_photons(sim_data &data) {
  // timer::stamp("emit_photons");
  auto &ptc = data.particles;
  auto &photons = data.photons;
  m_posInBlock.assign_dev(0, ptc.number());
  m_numPerBlock.assign_dev(0);
  m_cumNumPerBlock.assign_dev(0);
  auto data_p = get_data_ptrs(data);

  Kernels::count_photon_produced<particle_data>
      <<<m_blocksPerGrid, m_threadsPerBlock>>>(
          data_p, ptc.number(), m_numPerBlock.dev_ptr(),
          m_posInBlock.dev_ptr(), (curandState *)data.d_rand_states);
  CudaCheckError();

  thrust::device_ptr<int> ptrNumPerBlock(m_numPerBlock.dev_ptr());
  thrust::device_ptr<int> ptrCumNum(m_cumNumPerBlock.dev_ptr());

  // Scan the number of photons produced per block. The result gives
  // the offset for each block
  thrust::exclusive_scan(ptrNumPerBlock,
                         ptrNumPerBlock + m_blocksPerGrid, ptrCumNum);
  CudaCheckError();
  // Logger::print_debug("Scan finished");
  m_cumNumPerBlock.sync_to_host();
  m_numPerBlock.sync_to_host();
  int new_photons = m_cumNumPerBlock[m_blocksPerGrid - 1] +
                    m_numPerBlock[m_blocksPerGrid - 1];
  Logger::print_info("{} photons are produced!", new_photons);

  Kernels::produce_photons<particle_data, photon_data>
      <<<m_blocksPerGrid, m_threadsPerBlock>>>(
          data_p, ptc.number(), photons.number(),
          m_posInBlock.dev_ptr(), m_numPerBlock.dev_ptr(),
          m_cumNumPerBlock.dev_ptr(),
          (curandState *)data.d_rand_states);
  CudaCheckError();

  int padding = 100;
  photons.set_num(photons.number() + new_photons + padding);

  CudaSafeCall(cudaDeviceSynchronize());
  // timer::show_duration_since_stamp("Emitting photons", "ms",
  //                                  "emit_photons");
  // Logger::print_debug("Initialize finished");

  // Logger::print_info("There are {} photons in the pool",
  //                    photons.number());
  // cudaDeviceSynchronize();
}

void
radiative_transfer::produce_pairs(sim_data &data) {
  // timer::stamp("produce_pairs");
  auto &ptc = data.particles;
  auto &photons = data.photons;
  m_posInBlock.assign_dev(0, photons.number());
  m_numPerBlock.assign_dev(0);
  m_cumNumPerBlock.assign_dev(0);
  auto data_p = get_data_ptrs(data);

  Kernels::count_pairs_produced<photon_data>
      <<<m_blocksPerGrid, m_threadsPerBlock>>>(
          data_p, photons.number(), m_numPerBlock.dev_ptr(),
          m_posInBlock.dev_ptr(), (curandState *)data.d_rand_states);
  CudaCheckError();

  thrust::device_ptr<int> ptrNumPerBlock(m_numPerBlock.dev_ptr());
  thrust::device_ptr<int> ptrCumNum(m_cumNumPerBlock.dev_ptr());
  // cudaDeviceSynchronize();

  // Scan the number of photons produced per block. The last one will
  // be the total
  thrust::exclusive_scan(ptrNumPerBlock,
                         ptrNumPerBlock + m_blocksPerGrid, ptrCumNum);
  m_cumNumPerBlock.sync_to_host();
  m_numPerBlock.sync_to_host();
  int new_pairs = (m_cumNumPerBlock[m_blocksPerGrid - 1] +
                   m_numPerBlock[m_blocksPerGrid - 1]);
  Logger::print_info("{} electron-positron pairs are produced!",
                     new_pairs);

  Kernels::produce_pairs<particle_data, photon_data>
      <<<m_blocksPerGrid, m_threadsPerBlock>>>(
          data_p, photons.number(), ptc.number(),
          m_posInBlock.dev_ptr(), m_numPerBlock.dev_ptr(),
          m_cumNumPerBlock.dev_ptr(),
          (curandState *)data.d_rand_states);
  CudaCheckError();

  int padding = 100;
  ptc.set_num(ptc.number() + new_pairs * 2 + padding);
  // Logger::print_info("There are {} particles in the pool",
  //                    ptc.number());
  CudaSafeCall(cudaDeviceSynchronize());
  // timer::show_duration_since_stamp("Producing pairs", "ms",
  //                                  "produce_pairs");
}

}  // namespace Aperture