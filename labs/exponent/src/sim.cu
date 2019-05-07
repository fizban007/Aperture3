#include "cuda/constant_mem.h"
#include "cuda/core/cu_sim_environment.h"
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
add_new_particles(Scalar* ptc_E, size_t ptc_num, Scalar E, int num) {
  for (size_t idx = ptc_num + blockIdx.x * blockDim.x + threadIdx.x;
       idx < ptc_num + num; idx += blockDim.x * gridDim.x) {
    ptc_E[idx] = E;
  }
}

__global__ void
push_particles(Scalar* ptc_E, size_t ptc_num, Scalar Eacc, Scalar dt) {
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < ptc_num; idx += blockDim.x * gridDim.x) {
    Scalar E = ptc_E[idx];
    if (E >= 0.0f) {
      E += Eacc * dt;
      ptc_E[idx] = E;
    }
    if (idx == 0) {
      printf("0th particle has energy %f\n", E);
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
    if (idx == 0) {
      printf("0th photon has energy %f, path %f\n", E, ph_path[idx]);
    }
  }
}

__global__ void
exp_count_photons(Scalar* ptc_E, size_t ptc_num, int* ph_count,
                  int* ph_pos, curandState* states, Scalar dt) {
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
      if (tid == 0)
        printf("0th particle IC rate is %f\n", ic_rate * dt);
      float u = rng();
      if (u < ic_rate * dt) {
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
  CudaRng rng(&states[id]);
  for (uint32_t tid = id; tid < ptc_num;
       tid += blockDim.x * gridDim.x) {
    int pos_in_block = ph_pos[tid] - 1;
    Scalar E = ptc_E[tid];
    if (pos_in_block > -1 && E > 0.0f) {
      Scalar Eph = gen_photon_e(E, &states[id]);
      E -= Eph;
      ptc_E[tid] = max(E, 1.01);

      Scalar gg_rate = find_gg_rate(Eph);
      float u = rng();

      Scalar path = -(1.0f / gg_rate) * std::log(u);
      if (path > dev_params.delta_t * dev_params.max_steps)
        continue;
      int start_pos = ph_cum[blockIdx.x];
      int offset = ph_num + start_pos + pos_in_block;
      ph_E[offset] = Eph;
      // ph_E[offset] = -1.0;
      ph_path[offset] = path;
      if (tid == 0) {
        printf("emitting photon, gamma is %f, Eph is %f, path is %f\n",
               E + Eph, Eph, path);
      }
    }
  }
}

__global__ void
exp_count_gg_pairs(Scalar* ph_E, Scalar* ph_path, size_t ph_num,
                   int* pair_count, int* pair_pos, curandState* states,
                   Scalar dt) {
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
      ph_E[tid] = -1.0f;
    }
  }
}

__global__ void
exp_count_tpp_pairs(Scalar* ptc_E, size_t ptc_num, int* pair_count,
                    int* pair_pos, curandState* states, Scalar dt) {
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
      if (tid == 0)
        printf("0th particle tpp rate is %f\n", tpp_rate * dt);
      float u = rng();
      if (u < tpp_rate * dt) {
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
  curandState local_state = states[id];
  for (uint32_t tid = id; tid < ptc_num;
       tid += blockDim.x * gridDim.x) {
    int pos_in_block = pair_pos[tid] - 1;
    Scalar E = ptc_E[tid];
    if (pos_in_block > -1 && E > 0.0f) {
      // Scalar Em = find_tpp_Em(E);
      int start_pos = pair_cum[blockIdx.x];
      int offset = ptc_num + start_pos + pos_in_block;
      // if (tid == 0) printf("0th particle tpp Em is %f\n", Em * E);
      Scalar E_p = min(gen_tpp_Ep(E, &local_state), E - 2.02);
      Scalar E_m = min(gen_tpp_Ep(E, &local_state), E - E_p - 1.01);
      ptc_E[offset] = E_p;
      ptc_E[offset + 1] = E_m;
      ptc_E[tid] -= E_p + E_m;
      // ph_path[offset] = path;
    }
  }
  states[id] = local_state;
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
  m_ic.init(spec, spec.emin(), spec.emax(), n0);
  m_tpp.init(spec, spec.emin(), spec.emax(), n0);
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
  // ptc_E.sync_to_host();
  // for (int i = 0; i < num; i++) {
  //   ptc_E[ptc_num + i] = E;
  // }
  // ptc_E.sync_to_device();
  Kernels::add_new_particles<<<1, 128>>>(ptc_E.data_d(), ptc_num, E,
                                         num);
  CudaCheckError();
  ptc_num += num;
}

void
exponent_sim::prepare_initial_condition(int num_ptc, Scalar E) {
  for (int i = 0; i < ptc_E.size(); i++) {
    if (i < num_ptc) {
      ptc_E[i] = E;
    } else {
      ptc_E[i] = -1.0;
    }
  }
  ptc_E.sync_to_device();
  ptc_num = num_ptc;
  for (int i = 0; i < ph_E.size(); i++) {
    ph_E[i] = -1.0;
    ph_path[i] = -1.0;
  }
  ph_num = 0;
}

void
exponent_sim::produce_photons(Scalar dt) {
  m_pos_in_block.assign_dev(0, ph_num);
  m_num_per_block.assign_dev(0);
  m_cumnum_per_block.assign_dev(0);
  Kernels::
      exp_count_photons<<<m_blocks_per_grid, m_threads_per_block>>>(
          ptc_E.data_d(), ptc_num, m_num_per_block.data_d(),
          m_pos_in_block.data_d(), (curandState*)d_rand_states, dt);
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
exponent_sim::produce_pairs(Scalar dt) {
  int new_ptc;
  thrust::device_ptr<int> ptrNumPerBlock(m_num_per_block.data_d());
  thrust::device_ptr<int> ptrCumNum(m_cumnum_per_block.data_d());

  // Process gamma-gamma
  m_pos_in_block.assign_dev(0, ph_num);
  m_num_per_block.assign_dev(0);
  m_cumnum_per_block.assign_dev(0);
  Kernels::
      exp_count_gg_pairs<<<m_blocks_per_grid, m_threads_per_block>>>(
          ph_E.data_d(), ph_path.data_d(), ph_num,
          m_num_per_block.data_d(), m_pos_in_block.data_d(),
          (curandState*)d_rand_states, dt);
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
  Logger::print_info("{} particles are produced through gamma-gamma!",
                     new_ptc);

  Kernels::
      exp_emit_gg_pairs<<<m_blocks_per_grid, m_threads_per_block>>>(
          ptc_E.data_d(), ptc_num, ph_E.data_d(), ph_path.data_d(),
          ph_num, m_num_per_block.data_d(), m_pos_in_block.data_d(),
          m_cumnum_per_block.data_d(), (curandState*)d_rand_states);
  CudaCheckError();

  ptc_num += new_ptc;

  // Process tpp
  m_pos_in_block.assign_dev(0, ptc_num);
  m_num_per_block.assign_dev(0);
  m_cumnum_per_block.assign_dev(0);
  Kernels::
      exp_count_tpp_pairs<<<m_blocks_per_grid,
      m_threads_per_block>>>(
          ptc_E.data_d(), ptc_num, m_num_per_block.data_d(),
          m_pos_in_block.data_d(), (curandState*)d_rand_states, dt);
  CudaCheckError();

  // Scan the number of photons produced per block. The result gives
  // the offset for each block
  thrust::exclusive_scan(ptrNumPerBlock,
                         ptrNumPerBlock + m_blocks_per_grid,
                         ptrCumNum);
  CudaCheckError();

  m_cumnum_per_block.sync_to_host();
  m_num_per_block.sync_to_host();
  new_ptc = m_cumnum_per_block[m_blocks_per_grid - 1] +
            m_num_per_block[m_blocks_per_grid - 1];
  Logger::print_info("{} particles are produced through tpp!",
  new_ptc);

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

void
exponent_sim::sort_photons() {
  if (ph_num > 0) {
    thrust::device_ptr<Scalar> ptr_Eph(ph_E.data_d());
    thrust::device_ptr<Scalar> ptr_pathph(ph_path.data_d());

    auto z_end = thrust::remove_if(
        thrust::make_zip_iterator(
            thrust::make_tuple(ptr_Eph, ptr_pathph)),
        thrust::make_zip_iterator(
            thrust::make_tuple(ptr_Eph + ph_num, ptr_pathph + ph_num)),
        [] __host__ __device__(const thrust::tuple<Scalar, Scalar>& a) {
          return thrust::get<0>(a) < 2.0f;
        });
    auto end = z_end.get_iterator_tuple();
    auto Eph_end = thrust::get<0>(end);
    ph_num = Eph_end - ptr_Eph;
    Logger::print_info("After sort, there are {} photons in the pool",
                       ph_num);
  }
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
