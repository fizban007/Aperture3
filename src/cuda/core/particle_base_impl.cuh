#ifndef _PARTICLE_BASE_IMPL_CUH_
#define _PARTICLE_BASE_IMPL_CUH_

#include "cuda/constant_mem.h"
#include "cuda/cudaUtility.h"
// #include "cuda/data/particle_base_dev.h"
#include "core/particle_base.h"
#include "cuda/cuda_control.h"
#include "cuda/kernels.h"
#include "cuda/memory.h"
#include "utils/for_each_arg.hpp"
#include "utils/logger.h"
#include "utils/timer.h"

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/fill.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sort.h>

#include "visit_struct/visit_struct.hpp"
#include <algorithm>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
// #include "types/particles_dev.h"

namespace Aperture {

namespace Kernels {

struct assign_helper {
  size_t _nt, _n;

  HOST_DEVICE assign_helper(size_t nt, size_t n) : _nt(nt), _n(n) {}

  template <typename T>
  HD_INLINE void operator()(const char* name, const T& x1, T& x2) {
    x2[_nt] = x1[_n];
  }
};

template <typename T>
__global__ void
get_tracked_ptc_attr(T* ptr, uint32_t* flags, size_t num,
                     T* tracked_ptr, uint32_t* num_tracked) {
  for (size_t n = threadIdx.x + blockIdx.x * blockDim.x; n < num;
       n += blockDim.x * gridDim.x) {
    if (check_bit(flags[n], ParticleFlag::tracked)) {
      size_t nt = atomicAdd(num_tracked, 1u);
      tracked_ptr[nt] = ptr[n];
    }
  }
}

}  // namespace Kernels

// These are helper functors for for_each loops

struct assign_at_idx {
  size_t idx_;
  HOST_DEVICE assign_at_idx(size_t idx) : idx_(idx) {}

  template <typename T, typename U>
  HD_INLINE void operator()(T& t, const U& u) const {
    t[idx_] = u;
  }
};

struct fill_pos_amount {
  size_t pos_, amount_;

  fill_pos_amount(size_t pos, size_t amount)
      : pos_(pos), amount_(amount) {}

  template <typename T, typename U>
  void operator()(T& t, U& u) const {
    // std::fill_n(t + pos_, amount_, u);
    auto t_ptr = thrust::device_pointer_cast(t);
    thrust::fill_n(t_ptr + pos_, amount_, u);
    CudaCheckError();
  }
};

// struct sync_dev {
//   int devId_;
//   size_t size_;

//   sync_dev(int devId, size_t size) : devId_(devId), size_(size) {}

//   template <typename T>
//   void operator()(const char* name, T& x) const {
//     typedef typename std::remove_reference<decltype(*x)>::type
//     x_type; cudaMemPrefetchAsync(x, size_ * sizeof(x_type), devId_);
//   }
// };

struct copy_to_dest {
  size_t num_, src_pos_, dest_pos_;
  copy_to_dest(size_t num, size_t src_pos, size_t dest_pos)
      : num_(num), src_pos_(src_pos), dest_pos_(dest_pos) {}

  template <typename T, typename U>
  void operator()(T& t, U& u) const {
    auto t_ptr = thrust::device_pointer_cast(t);
    auto u_ptr = thrust::device_pointer_cast(u);
    thrust::copy_n(u_ptr + src_pos_, num_, t_ptr + dest_pos_);
    // std::copy(u + src_pos_, u + src_pos_ + num_, t + dest_pos_);
  }
};

// template <typename ArrayType>
struct rearrange_array {
  // ArrayType& array_;
  // thrust::device_ptr<Index_t>& index_;
  Index_t* index_;
  size_t N_;
  void* tmp_ptr_;
  std::string skip_;

  rearrange_array(Index_t* index, size_t N, void* tmp_ptr,
                  const std::string& skip)
      : index_(index), N_(N), tmp_ptr_(tmp_ptr), skip_(skip) {}

  template <typename T, typename U>
  void operator()(const char* name, T& x, U& u) const {
    auto ptr_index = thrust::device_pointer_cast(index_);
    // Logger::print_info("rearranging {}", name);
    if (std::strcmp(name, skip_.c_str()) == 0) {
      // Logger::print_info("skipping {}", name);
      return;
    }
    auto x_ptr = thrust::device_pointer_cast(x);
    auto tmp_ptr =
        thrust::device_pointer_cast(reinterpret_cast<U*>(tmp_ptr_));
    thrust::gather(ptr_index, ptr_index + N_, x_ptr, tmp_ptr);
    thrust::copy_n(tmp_ptr, N_, x_ptr);
    CudaCheckError();
  }
};

template <typename ParticleClass>
particle_base<ParticleClass>::particle_base()
    : m_tmp_data_ptr(nullptr), m_index(nullptr) {
  visit_struct::for_each(
      m_data, [](const char* name, auto& x) { x = nullptr; });
}

template <typename ParticleClass>
particle_base<ParticleClass>::particle_base(std::size_t max_num) {
  std::cout << "New particle array with size " << max_num << std::endl;
  alloc_mem(max_num);
  // auto alloc = alloc_cuda_managed(max_num);
  // auto alloc = alloc_cuda_device(max_num);
  // alloc("index", m_index);
  cudaMalloc(&m_index, max_num * sizeof(Index_t));
  cudaMalloc(&m_tmp_data_ptr, max_num * sizeof(double));
  // m_index.resize(max_num, 0);
  // m_index_bak.resize(max_num, 0);
  initialize();
}

template <typename ParticleClass>
particle_base<ParticleClass>::particle_base(
    const particle_base<ParticleClass>& other) {
  std::size_t n = other.m_size;
  m_size = n;
  m_number = other.m_number;

  alloc_mem(n);
  // auto alloc = alloc_cuda_managed(n);
  // auto alloc = alloc_cuda_device(n);
  // alloc(m_index);
  cudaMalloc(&m_index, n * sizeof(Index_t));
  cudaMalloc(&m_tmp_data_ptr, n * sizeof(double));
  // alloc((double*)m_tmp_data_ptr);
  // m_index.resize(n);
  // m_index_bak.resize(n);
  copy_from(other, n);
}

template <typename ParticleClass>
particle_base<ParticleClass>::particle_base(
    particle_base<ParticleClass>&& other) {
  m_size = other.m_size;
  m_number = other.m_number;

  // m_data_ptr = other.m_data_ptr;
  m_data = other.m_data;
  // auto alloc = alloc_cuda_managed(m_size);
  // auto alloc = alloc_cuda_device(m_size);
  // alloc(m_index);
  cudaMalloc(&m_tmp_data_ptr, m_size * sizeof(double));
  cudaMalloc(&m_index, m_size * sizeof(Index_t));
  // alloc((double*)m_tmp_data_ptr);
  // m_index.resize(other.m_size);
  // m_index_bak.resize(other.m_size);

  // boost::fusion::for_each(other.m_data, set_nullptr());
  visit_struct::for_each(
      other.m_data, [](const char* name, auto& x) { x = nullptr; });
  // other.m_data_ptr = nullptr;
}

template <typename ParticleClass>
particle_base<ParticleClass>::~particle_base() {
  free_mem();
  free_cuda()(m_index);
  free_cuda()(m_tmp_data_ptr);
  // CudaSafeCall(cudaFree(m_index));
  // CudaSafeCall(cudaFree(m_tmp_data_ptr));
}

template <typename ParticleClass>
void
particle_base<ParticleClass>::alloc_mem(std::size_t max_num,
                                        std::size_t alignment) {
  alloc_struct_of_arrays(m_data, max_num);
  alloc_struct_of_arrays_managed(m_tracked, MAX_TRACKED);
}

template <typename ParticleClass>
void
particle_base<ParticleClass>::free_mem() {
  free_struct_of_arrays(m_data);
  free_struct_of_arrays(m_tracked);
}

template <typename ParticleClass>
void
particle_base<ParticleClass>::initialize() {
  erase(0, m_size);
  m_number = 0;
}

template <typename ParticleClass>
void
particle_base<ParticleClass>::resize(std::size_t max_num) {
  m_size = max_num;
  if (m_number > max_num) m_number = max_num;
  free_mem();
  alloc_mem(max_num);

  free_cuda()(m_index);
  free_cuda()(m_tmp_data_ptr);
  // auto alloc = alloc_cuda_managed(max_num);
  // auto alloc = alloc_cuda_device(max_num);
  // alloc(m_index);
  cudaMalloc(&m_index, max_num * sizeof(Index_t));
  cudaMalloc(&m_tmp_data_ptr, max_num * sizeof(double));
  // alloc((double*)m_tmp_data_ptr);

  // m_index.resize(max_num, 0);
  // m_index_bak.resize(max_num, 0);
  initialize();
}

template <typename ParticleClass>
void
particle_base<ParticleClass>::erase(std::size_t pos,
                                    std::size_t amount) {
  if (pos + amount > m_size) amount = m_size - pos;
  std::cout << "Erasing from index " << pos << " for " << amount
            << " number of particles" << std::endl;

  auto ptc = ParticleClass{};
  for_each_arg(m_data, ptc, fill_pos_amount{pos, amount});
}

template <typename ParticleClass>
void
particle_base<ParticleClass>::put(Index_t pos,
                                  const ParticleClass& part) {
  if (pos >= m_size)
    throw std::runtime_error(
        "Trying to insert particle beyond the end of the array."
        "Resize it first!");

  for_each_arg(m_data, part, assign_at_idx(pos));
  if (pos >= m_number) m_number = pos + 1;
}

template <typename ParticleClass>
void
particle_base<ParticleClass>::append(const ParticleClass& part) {
  // put(m_number, x, p, cell, flag);
  put(m_number, part);
}

template <typename ParticleClass>
void
particle_base<ParticleClass>::copy_from(
    const particle_base<ParticleClass>& other, std::size_t num,
    std::size_t src_pos, std::size_t dest_pos) {
  // Adjust the number so that we don't over fill
  if (dest_pos + num > m_size) num = m_size - dest_pos;
  for_each_arg(m_data, other.m_data,
               copy_to_dest(num, src_pos, dest_pos));
  // Adjust the new number of particles in the array
  if (dest_pos + num > m_number) m_number = dest_pos + num;
}

template <typename ParticleClass>
void
particle_base<ParticleClass>::sort_by_cell(const Grid& grid) {
  if (m_number > 0) {
    // Generate particle index array
    auto ptr_cell = thrust::device_pointer_cast(m_data.cell);
    auto ptr_idx = thrust::device_pointer_cast(m_index);
    thrust::counting_iterator<Index_t> iter(0);
    thrust::copy_n(iter, this->m_number, ptr_idx);

    // Sort the index array by key
    thrust::sort_by_key(ptr_cell, ptr_cell + this->m_number, ptr_idx);
    // cudaDeviceSynchronize();
    Logger::print_debug("Finished sorting");

    // Move the rest of particle array using the new index
    rearrange_arrays("cell");

    // Update the new number of particles
    const int padding = 100;
    m_number =
        thrust::upper_bound(ptr_cell, ptr_cell + m_number + padding,
                            MAX_CELL - 1) -
        ptr_cell;

    // Logger::print_info("Sorting complete, there are {} particles in
    // the pool", m_number); cudaDeviceSynchronize();
    CudaCheckError();
  }
}

template <typename ParticleClass>
void
particle_base<ParticleClass>::rearrange_arrays(
    const std::string& skip) {
  const uint32_t padding = 100;
  auto ptc = ParticleClass();
  for_each_arg_with_name(
      m_data, ptc,
      rearrange_array{m_index, std::min(m_size, m_number + padding),
                      m_tmp_data_ptr, skip});
}

template <typename ParticleClass>
void
particle_base<ParticleClass>::clear_guard_cells(const Grid& grid) {
  erase_ptc_in_guard_cells(m_data.cell, m_number);
}

template <typename ParticleClass>
void
particle_base<ParticleClass>::get_tracked_ptc() {
  uint32_t* num_tracked;
  CudaSafeCall(cudaMalloc(&num_tracked, sizeof(uint32_t)));
  CudaSafeCall(cudaMemset(num_tracked, 0, sizeof(uint32_t)));

  visit_struct::for_each(
      m_data, m_tracked,
      [this, &num_tracked](const char* name, auto& u, auto& v) {
        CudaSafeCall(cudaMemset(num_tracked, 0, sizeof(uint32_t)));
        Kernels::get_tracked_ptc_attr<<<512, 512>>>(
            u, m_data.flag, m_number, v, num_tracked);
        CudaCheckError();
        CudaSafeCall(cudaDeviceSynchronize());
      });
  CudaSafeCall(cudaMemcpy(&m_num_tracked, num_tracked, sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaFree(num_tracked));
}

// template <typename ParticleClass>
// void
// particle_base<ParticleClass>::compute_spectrum(
//     int num_bins, std::vector<Scalar>& energies,
//     std::vector<uint32_t>& nums) {
//   // Assume the particle energies have been computed
//   energies.resize(num_bins, 0.0);
//   nums.resize(num_bins, 0);

//   // Find maximum energy in the array now
//   thrust::device_ptr<Scalar> E_ptr =
//       thrust::device_pointer_cast(m_data.E);
//   Scalar E_max = *thrust::max_element(E_ptr, E_ptr + m_number);
//   // Logger::print_info("Maximum energy is {}", E_max);

//   // Partition the energy bin up to max energy times a factor
//   Scalar dlogE = std::log(E_max) / (Scalar)num_bins;
//   for (int i = 0; i < num_bins; i++) {
//     energies[i] = std::exp((0.5f + (Scalar)i) * dlogE);
//     // Logger::print_info("{}", energies[i]);
//   }

//   // Do a histogram
//   uint32_t* d_energies;
//   cudaMalloc(&d_energies, num_bins * sizeof(uint32_t));
//   thrust::device_ptr<uint32_t> ptr_energies =
//       thrust::device_pointer_cast(d_energies);
//   thrust::fill_n(ptr_energies, num_bins, 0);
//   // cudaDeviceSynchronize();

//   compute_energy_histogram(d_energies, m_data.E, m_number, num_bins,
//                            E_max);

//   // Copy the resulting histogram to output
//   cudaMemcpy(nums.data(), d_energies, num_bins * sizeof(uint32_t),
//              cudaMemcpyDeviceToHost);

//   cudaFree(d_energies);
// }

}  // namespace Aperture

#endif  // _PARTICLE_BASE_IMPL_CUH_
