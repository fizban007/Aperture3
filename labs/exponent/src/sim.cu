#include "cuda/constant_mem.h"
#include "cuda/core/sim_environment_dev.h"
#include "cuda/cudaUtility.h"
#include "cuda/cudarng.h"
#include "cuda/kernels.h"
#include "cuda/radiation/rt_ic_dev.h"
#include "cuda/radiation/rt_tpp_dev.h"
#include "radiation/spectra.h"
#include "sim.h"
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

namespace Aperture {

namespace Kernels {

__global__ void
push_particles(Scalar* ptc_E, size_t ptc_num, Scalar Eacc, Scalar dt) {
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < ptc_num; idx += blockDim.x * gridDim.x) {
    Scalar E = ptc_E[idx];
    if (E > 0.0f) {
      E += Eacc * dt;
      ptc_E[idx] = E;
    }
  }
}

__global__ void
push_photons(Scalar* ph_E, Scalar* ph_path, size_t ph_num, Scalar dt) {
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < ph_num;
       idx += blockDim.x * gridDim.x) {
    Scalar E = ph_E[idx];
    if (E > 0.0f) {
      ph_path[idx] -= dt;
    }
  }
}

__global__ void
exp_count_photons(Scalar* ptc_E, size_t ptc_num, int* ph_count,
                  int* ph_pos, curandState* states) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  CudaRng rng(&states[id]);
  __shared__ int photonProduced;
  if (threadIdx.x == 0) photonProduced = 0;
  __syncthreads();

  for (uint32_t tid = id; tid < ptc_num;
       tid += blockDim.x * gridDim.x) {
    Scalar E = ptc_E[tid];
    if (E > 0.0f) {
      // TODO: Multiply by some normalization constant for ic_rate
      Scalar ic_rate = find_ic_rate(E);
      float u = rng();
      if (u < ic_rate) {
        ph_pos[tid] = atomicAdd(&photonProduced, 1) + 1;
      }
    }
  }

  __syncthreads();

  // Record the number of photons produced this block to global array
  if (threadIdx.x == 0) {
    ph_count[blockIdx.x] = photonProduced;
  }
}

__global__ void
exp_emit_photons(Scalar* ptc_E, size_t ptc_num, Scalar* ph_E,
                 Scalar* ph_path, size_t ph_num, int* ph_count,
                 int* ph_pos, int* ph_cum, curandState* states) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  // CudaRng rng(&states[id]);
  for (uint32_t tid = id; tid < ptc_num;
       tid += blockDim.x * gridDim.x) {
    int pos_in_block = ph_pos[tid] - 1;
    Scalar E = ptc_E[tid];
    if (pos_in_block > -1 && E > 0.0f) {
      Scalar Eph = gen_photon_e(E, &states[id]);
      E -= Eph;
      ptc_E[tid] = E;

      Scalar gg_rate = find_gg_rate(Eph);
      // TODO: normalize gg_rate and get free path
      Scalar path = 1.0f / gg_rate;
      if (path > dev_params.r_cutoff) continue;
      int start_pos = ph_cum[blockIdx.x];
      int offset = ph_num + start_pos + pos_in_block;
      ph_E[offset] = Eph;
      ph_path[offset] = path;
    }
  }
}

__global__ void
exp_count_gg_pairs(Scalar* ph_E, Scalar* ph_path, size_t ph_num,
                   int* pair_count, int* pair_pos,
                   curandState* states) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  CudaRng rng(&states[id]);
  __shared__ int pairProduced;
  if (threadIdx.x == 0) pairProduced = 0;
  __syncthreads();

  for (uint32_t tid = id; tid < ph_num; tid += blockDim.x * gridDim.x) {
    Scalar E = ph_E[tid];
    Scalar path = ph_path[tid];
    if (E > 0.0f && path <= 0.0f) {
      pair_pos[tid] = atomicAdd(&pairProduced, 2) + 2;
    }
  }

  __syncthreads();

  // Record the number of pairs produced this block to global array
  if (threadIdx.x == 0) {
    pair_count[blockIdx.x] = pairProduced;
  }
}

__global__ void
exp_emit_gg_pairs(Scalar* ptc_E, size_t ptc_num, Scalar* ph_E,
                  Scalar* ph_path, size_t ph_num, int* pair_count,
                  int* pair_pos, int* pair_cum, curandState* states) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  // CudaRng rng(&states[id]);
  for (uint32_t tid = id; tid < ptc_num;
       tid += blockDim.x * gridDim.x) {
    int pos_in_block = pair_pos[tid] - 1;
    Scalar E = ph_E[tid];
    if (pos_in_block > -1 && E > 0.0f) {
      int start_pos = pair_cum[blockIdx.x];
      int offset = ph_num + start_pos + pos_in_block;
      ptc_E[offset] = 0.5f * E;
      ptc_E[offset + 1] = 0.5f * E;
      // ph_path[offset] = path;
    }
  }
}

__global__ void
exp_count_tpp_pairs(Scalar* ptc_E, size_t ptc_num, int* pair_count,
                    int* pair_pos, curandState* states) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  CudaRng rng(&states[id]);
  __shared__ int pairProduced;
  if (threadIdx.x == 0) pairProduced = 0;
  __syncthreads();

  for (uint32_t tid = id; tid < ptc_num;
       tid += blockDim.x * gridDim.x) {
    Scalar E = ptc_E[tid];
    // Scalar path = ph_path[tid];
    if (E > 0.0f) {
      // TODO: Multiply by some normalization constant for ic_rate
      Scalar tpp_rate = find_tpp_rate(E);
      float u = rng();
      if (u < tpp_rate) {
        pair_pos[tid] = atomicAdd(&pairProduced, 2) + 2;
      }
    }
  }

  __syncthreads();

  // Record the number of pairs produced this block to global array
  if (threadIdx.x == 0) {
    pair_count[blockIdx.x] = pairProduced;
  }
}

__global__ void
exp_emit_tpp_pairs(Scalar* ptc_E, size_t ptc_num, int* pair_count,
                   int* pair_pos, int* pair_cum, curandState* states) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  // CudaRng rng(&states[id]);
  for (uint32_t tid = id; tid < ptc_num;
       tid += blockDim.x * gridDim.x) {
    int pos_in_block = pair_pos[tid] - 1;
    Scalar E = ptc_E[tid];
    if (pos_in_block > -1 && E > 0.0f) {
      Scalar Em = find_tpp_Em(E);
      int start_pos = pair_cum[blockIdx.x];
      int offset = ptc_num + start_pos + pos_in_block;
      ptc_E[offset] = Em;
      ptc_E[offset + 1] = Em;
      ptc_E[tid] -= 2.0f * Em;
      // ph_path[offset] = path;
    }
  }
}

__global__ void
exp_compute_spectrum(Scalar* E, size_t num, Scalar* ne) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  // CudaRng rng(&states[id]);
  for (uint32_t tid = id; tid < num; tid += blockDim.x * gridDim.x) {
    Scalar Ep = E[tid];
    if (Ep > 0.0f) {
      int n = find_n_gamma(Ep);
      atomicAdd(&ne[n], 1.0f);
    }
  }
}

}  // namespace Kernels

exponent_sim::exponent_sim(cu_sim_environment& env)
    : m_env(env),
      m_ic(env.params()),
      m_tpp(env.params()),
      ptc_num(0),
      ph_num(0),
      ptc_E(env.params().max_ptc_number),
      ph_E(env.params().max_photon_number),
      ph_path(env.params().max_photon_number),
      ptc_spec(env.params().n_gamma),
      ph_spec(env.params().n_gamma),
      m_threads_per_block(512),
      m_blocks_per_grid(256),
      m_num_per_block(m_blocks_per_grid),
      m_cumnum_per_block(m_blocks_per_grid),
      m_pos_in_block(env.params().max_ptc_number) {
  ptc_E.assign_dev(-1.0);
  ph_E.assign_dev(-1.0);
  ph_path.assign_dev(0.0);
  ptc_spec.assign_dev(0.0);
  ph_spec.assign_dev(0.0);

  int seed = env.params().random_seed;
  CudaSafeCall(cudaMalloc(
      &d_rand_states,
      m_threads_per_block * m_blocks_per_grid * sizeof(curandState)));
  init_rand_states((curandState*)d_rand_states, seed,
                   m_threads_per_block, m_blocks_per_grid);
}

exponent_sim::~exponent_sim() {
  CudaSafeCall(cudaFree((curandState*)d_rand_states));
}

template <typename T>
void
exponent_sim::init_spectra(const T& spec, double n0) {
  m_ic.init(spec, spec.emin(), spec.emax());
  m_tpp.init(spec, spec.emin(), spec.emax());
}

void
exponent_sim::push_particles(Scalar Eacc, Scalar dt) {
  Kernels::push_particles<<<256, 512>>>(ptc_E.data_d(), ptc_num, Eacc,
                                        dt);
  CudaCheckError();
  Kernels::push_photons<<<256, 512>>>(ph_E.data_d(), ph_path.data_d(),
                                      ph_num, dt);
  CudaCheckError();
}

void
exponent_sim::add_new_particles(int num, Scalar E) {
  ptc_E.sync_to_host();
  for (int i = 0; i < num; i++) {
    ptc_E[ptc_num + i] = E;
  }
  ptc_num += num;
}

void
exponent_sim::produce_photons() {
  m_pos_in_block.assign_dev(0, ph_num);
  m_num_per_block.assign_dev(0);
  m_cumnum_per_block.assign_dev(0);
  Kernels::
      exp_count_photons<<<m_blocks_per_grid, m_threads_per_block>>>(
          ptc_E.data_d(), ptc_num, m_num_per_block.data_d(),
          m_pos_in_block.data_d(), (curandState*)d_rand_states);
  CudaCheckError();

  thrust::device_ptr<int> ptrNumPerBlock(m_num_per_block.data_d());
  thrust::device_ptr<int> ptrCumNum(m_cumnum_per_block.data_d());

  // Scan the number of photons produced per block. The result gives
  // the offset for each block
  thrust::exclusive_scan(ptrNumPerBlock,
                         ptrNumPerBlock + m_blocks_per_grid, ptrCumNum);
  CudaCheckError();

  m_cumnum_per_block.sync_to_host();
  m_num_per_block.sync_to_host();
  int new_photons = m_cumnum_per_block[m_blocks_per_grid - 1] +
                    m_num_per_block[m_blocks_per_grid - 1];
  Logger::print_info("{} photons are produced!", new_photons);

  Kernels::exp_emit_photons<<<m_blocks_per_grid, m_threads_per_block>>>(
      ptc_E.data_d(), ptc_num, ph_E.data_d(), ph_path.data_d(), ph_num,
      m_num_per_block.data_d(), m_pos_in_block.data_d(),
      m_cumnum_per_block.data_d(), (curandState*)d_rand_states);
  CudaCheckError();

  ph_num += new_photons;
}

void
exponent_sim::produce_pairs() {
  m_pos_in_block.assign_dev(0, ph_num);
  m_num_per_block.assign_dev(0);
  m_cumnum_per_block.assign_dev(0);
  Kernels::
      exp_count_gg_pairs<<<m_blocks_per_grid, m_threads_per_block>>>(
          ph_E.data_d(), ph_path.data_d(), ph_num,
          m_num_per_block.data_d(), m_pos_in_block.data_d(),
          (curandState*)d_rand_states);
  CudaCheckError();

  thrust::device_ptr<int> ptrNumPerBlock(m_num_per_block.data_d());
  thrust::device_ptr<int> ptrCumNum(m_cumnum_per_block.data_d());

  // Scan the number of photons produced per block. The result gives
  // the offset for each block
  thrust::exclusive_scan(ptrNumPerBlock,
                         ptrNumPerBlock + m_blocks_per_grid, ptrCumNum);
  CudaCheckError();

  m_cumnum_per_block.sync_to_host();
  m_num_per_block.sync_to_host();
  int new_ptc = m_cumnum_per_block[m_blocks_per_grid - 1] +
                m_num_per_block[m_blocks_per_grid - 1];
  Logger::print_info("{} particles are produced through gamma-gamma!",
                     new_ptc);

  Kernels::
      exp_emit_gg_pairs<<<m_blocks_per_grid, m_threads_per_block>>>(
          ptc_E.data_d(), ptc_num, ph_E.data_d(), ph_path.data_d(),
          ph_num, m_num_per_block.data_d(), m_pos_in_block.data_d(),
          m_cumnum_per_block.data_d(), (curandState*)d_rand_states);
  CudaCheckError();

  ptc_num += new_ptc;
  m_pos_in_block.assign_dev(0, ph_num);
  m_num_per_block.assign_dev(0);
  m_cumnum_per_block.assign_dev(0);
  Kernels::
      exp_count_tpp_pairs<<<m_blocks_per_grid, m_threads_per_block>>>(
          ptc_E.data_d(), ptc_num, m_num_per_block.data_d(),
          m_pos_in_block.data_d(), (curandState*)d_rand_states);
  CudaCheckError();

  // Scan the number of photons produced per block. The result gives
  // the offset for each block
  thrust::exclusive_scan(ptrNumPerBlock,
                         ptrNumPerBlock + m_blocks_per_grid, ptrCumNum);
  CudaCheckError();

  m_cumnum_per_block.sync_to_host();
  m_num_per_block.sync_to_host();
  new_ptc = m_cumnum_per_block[m_blocks_per_grid - 1] +
            m_num_per_block[m_blocks_per_grid - 1];
  Logger::print_info("{} particles are produced through tpp!", new_ptc);

  Kernels::
      exp_emit_tpp_pairs<<<m_blocks_per_grid, m_threads_per_block>>>(
          ptc_E.data_d(), ptc_num, m_num_per_block.data_d(),
          m_pos_in_block.data_d(), m_cumnum_per_block.data_d(),
          (curandState*)d_rand_states);
  CudaCheckError();

  ptc_num += new_ptc;
}

void
exponent_sim::compute_spectrum() {
  ptc_spec.assign_dev(0.0);
  ph_spec.assign_dev(0.0);
  Kernels::
      exp_compute_spectrum<<<m_blocks_per_grid, m_threads_per_block>>>(
          ptc_E.data_d(), ptc_num, ptc_spec.data_d());
  CudaCheckError();

  Kernels::
      exp_compute_spectrum<<<m_blocks_per_grid, m_threads_per_block>>>(
          ph_E.data_d(), ph_num, ph_spec.data_d());
  CudaCheckError();
}

// Instantiate templates
template void exponent_sim::init_spectra<Spectra::black_body>(
    const Spectra::black_body& spec, double n0);
template void exponent_sim::init_spectra<Spectra::power_law_hard>(
    const Spectra::power_law_hard& spec, double n0);
template void exponent_sim::init_spectra<Spectra::power_law_soft>(
    const Spectra::power_law_soft& spec, double n0);
template void exponent_sim::init_spectra<Spectra::mono_energetic>(
    const Spectra::mono_energetic& spec, double n0);
template void exponent_sim::init_spectra<Spectra::broken_power_law>(
    const Spectra::broken_power_law& spec, double n0);

}  // namespace Aperture