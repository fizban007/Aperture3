#ifndef _PARTICLE_BASE_IMPL_H_
#define _PARTICLE_BASE_IMPL_H_

#include "data/particle_base.h"

#include "utils/for_each_arg.hpp"
#include "utils/logger.h"
#include "utils/memory.h"
#include "utils/timer.h"

#include "visit_struct/visit_struct.hpp"
#include <algorithm>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace Aperture {

template <typename ParticleClass>
particle_base<ParticleClass>::particle_base()
    : m_numMax(0),
      m_number(0),
      m_tmp_data_ptr(nullptr),
      m_index(nullptr) {
  visit_struct::for_each(
      m_data, [](const char* name, auto& x) { x = nullptr; });
}

template <typename ParticleClass>
particle_base<ParticleClass>::particle_base(std::size_t max_num)
    : m_numMax(max_num), m_number(0) {
  std::cout << "New particle array with size " << max_num << std::endl;
  alloc_mem(max_num);
  // auto alloc = alloc_cuda_managed(max_num);
  // auto alloc = alloc_cuda_device(max_num);
  // alloc(m_index);
  // cudaMalloc(&m_tmp_data_ptr, max_num * sizeof(double));
  // m_index.resize(max_num, 0);
  // m_index_bak.resize(max_num, 0);

  // TODO: Allocate m_index and m_tmp_data_ptr
  initialize();
}

template <typename ParticleClass>
particle_base<ParticleClass>::particle_base(
    const particle_base<ParticleClass>& other) {
  std::size_t n = other.m_numMax;
  m_numMax = n;
  m_number = other.m_number;
  // m_sorted = other.m_sorted;

  alloc_mem(n);
  // auto alloc = alloc_cuda_managed(n);
  // auto alloc = alloc_cuda_device(n);
  // alloc(m_index);
  // cudaMalloc(&m_tmp_data_ptr, n * sizeof(double));
  // alloc((double*)m_tmp_data_ptr);
  // m_index.resize(n);
  // m_index_bak.resize(n);
  copy_from(other, n);
}

template <typename ParticleClass>
particle_base<ParticleClass>::particle_base(
    particle_base<ParticleClass>&& other) {
  m_numMax = other.m_numMax;
  m_number = other.m_number;
  // m_sorted = other.m_sorted;

  // m_data_ptr = other.m_data_ptr;
  m_data = other.m_data;
  // auto alloc = alloc_cuda_managed(m_numMax);
  // auto alloc = alloc_cuda_device(m_numMax);
  // alloc(m_index);
  // cudaMalloc(&m_tmp_data_ptr, m_numMax * sizeof(double));
  // alloc((double*)m_tmp_data_ptr);
  // m_index.resize(other.m_numMax);
  // m_index_bak.resize(other.m_numMax);

  // boost::fusion::for_each(other.m_data, set_nullptr());
  visit_struct::for_each(
      other.m_data, [](const char* name, auto& x) { x = nullptr; });
}

template <typename ParticleClass>
particle_base<ParticleClass>::~particle_base() {
  free_mem();
  // free_cuda()(m_index);
  // free_cuda()(m_tmp_data_ptr);
}

template <typename ParticleClass>
void
particle_base<ParticleClass>::alloc_mem(std::size_t max_num) {
  // alloc_struct_of_arrays(m_data, max_num);
  visit_struct::for_each(m_data, [max_num](const char* name, auto& x) {
    typedef typename std::remove_reference<decltype(*x)>::type x_type;
    void* p = aligned_malloc(max_num * sizeof(x_type), 64);
    x = reinterpret_cast<
        typename std::remove_reference<decltype(x)>::type>(p);
  });
}

template <typename ParticleClass>
void
particle_base<ParticleClass>::free_mem() {
  visit_struct::for_each(m_data, [](const char* name, auto& x) {
    if (x != nullptr) {
      aligned_free(x);
      x = nullptr;
    }
  });
}

template <typename ParticleClass>
void
particle_base<ParticleClass>::initialize() {
  erase(0, m_numMax);
  m_number = 0;
}

template <typename ParticleClass>
void
particle_base<ParticleClass>::resize(std::size_t max_num) {
  m_numMax = max_num;
  if (m_number > max_num) m_number = max_num;
  free_mem();
  alloc_mem(max_num);

  // free_cuda()(m_index);
  // free_cuda()(m_tmp_data_ptr);
  // auto alloc = alloc_cuda_managed(max_num);
  // auto alloc = alloc_cuda_device(max_num);
  // alloc(m_index);
  // cudaMalloc(&m_tmp_data_ptr, max_num * sizeof(double));
  // alloc((double*)m_tmp_data_ptr);

  // m_index.resize(max_num, 0);
  // m_index_bak.resize(max_num, 0);
  initialize();
}

template <typename ParticleClass>
void
particle_base<ParticleClass>::erase(std::size_t pos,
                                   std::size_t amount) {
  if (pos + amount > m_numMax) amount = m_numMax - pos;
  std::cout << "Erasing from " << pos << " for " << amount
            << " number of particles" << std::endl;

  auto ptc = ParticleClass{};
  // for_each_arg(m_data, ptc, fill_pos_amount{pos, amount});
  std::fill_n(m_data.cell + pos, amount, MAX_CELL);
}

// template <typename ParticleClass>
// void
// particle_base<ParticleClass>::put(Index_t pos,
//                                  const ParticleClass& part) {
//   if (pos >= m_numMax)
//     throw std::runtime_error(
//         "Trying to insert particle beyond the end of the array.
//         Resize " "it " "first!");

//   for_each_arg(m_data, part, assign_at_idx(pos));
//   if (pos >= m_number) m_number = pos + 1;
// }

// template <typename ParticleClass>
// void
// particle_base<ParticleClass>::swap(Index_t pos, ParticleClass& part) {
//   ParticleClass p_tmp = m_data[pos];
//   if (pos >= m_numMax)
//     throw std::runtime_error(
//         "Trying to swap particle beyond the end of the array. Resize
//         " "it " "first!");

//   // typedef boost::fusion::vector<array_type&, const ParticleClass&>
//   // seq;
//   // boost::fusion::for_each(boost::fusion::zip_view<seq>(seq(m_data,
//   // part)),
//   //                         assign_at_idx(pos) );
//   for_each_arg(m_data, part, assign_at_idx(pos));
//   part = p_tmp;
//   if (pos >= m_number) m_number = pos + 1;
// }

// template <typename ParticleClass>
// void
// particle_base<ParticleClass>::append(const ParticleClass& part) {
//   // put(m_number, x, p, cell, flag);
//   put(m_number, part);
// }

template <typename ParticleClass>
void
particle_base<ParticleClass>::copy_from(
    const particle_base<ParticleClass>& other, std::size_t num,
    std::size_t src_pos, std::size_t dest_pos) {
  // Adjust the number so that we don't over fill
  if (dest_pos + num > m_numMax) num = m_numMax - dest_pos;
  for_each_arg(m_data, other.m_data,
               [num, src_pos, dest_pos](auto& t, auto& u) {
                 std::copy_n(u + src_pos, num, t + dest_pos);
               });
  // Adjust the new number of particles in the array
  if (dest_pos + num > m_number) m_number = dest_pos + num;
}

// template <typename ParticleClass>
// void
// particle_base<ParticleClass>::compute_tile_num() {
//   compute_tile(m_data.tile, m_data.cell, m_number);
// }

// template <typename ParticleClass>
// void
// particle_base<ParticleClass>::sort_by_tile() {
//   // First compute the tile number according to current cell id
//   compute_tile(m_data.tile, m_data.cell, m_number);

//   // Generate particle index array
//   auto ptr_tile = thrust::device_pointer_cast(m_data.tile);
//   auto ptr_idx = thrust::device_pointer_cast(m_index);
//   thrust::counting_iterator<Index_t> iter(0);
//   thrust::copy_n(iter, this->m_number, ptr_idx);

//   // Sort the index array by key
//   thrust::sort_by_key(ptr_tile, ptr_tile + this->m_number, ptr_idx);

//   // Move the rest of particle array using the new index
//   rearrange_arrays("tile");

//   // Update the new number of particles
//   const int padding = 100;
//   m_number =
//       thrust::upper_bound(ptr_tile, ptr_tile + m_number + padding,
//                           MAX_TILE - 1) -
//       ptr_tile;

//   // Wait for GPU to finish before accessing on host
//   cudaDeviceSynchronize();
// }

template <typename ParticleClass>
void
particle_base<ParticleClass>::sort_by_cell() {
  // Generate particle index array
  // auto ptr_cell = thrust::device_pointer_cast(m_data.cell);
  // auto ptr_idx = thrust::device_pointer_cast(m_index);
  // thrust::counting_iterator<Index_t> iter(0);
  // thrust::copy_n(iter, this->m_number, ptr_idx);

  // Sort the index array by key
  // thrust::sort_by_key(ptr_cell, ptr_cell + this->m_number, ptr_idx);
  // cudaDeviceSynchronize();

  // Move the rest of particle array using the new index
  rearrange_arrays("cell");

  // Update the new number of particles
  // const int padding = 100;
  // m_number =
  //     thrust::upper_bound(ptr_cell, ptr_cell + m_number + padding,
  //                         MAX_CELL - 1) -
  //     ptr_cell;

  // Logger::print_info("Sorting complete, there are {} particles in the
  // pool", m_number);

  // TODO: implement sorting
}

template <typename ParticleClass>
void
particle_base<ParticleClass>::rearrange_arrays(const std::string& skip) {
  const uint32_t padding = 100;
  auto ptc = ParticleClass();
  // for_each_arg_with_name(
  //     m_data, ptc,
  //     rearrange_array{m_index, std::min(m_numMax, m_number +
  //     padding),
  //                     m_tmp_data_ptr, skip});
}

template <typename ParticleClass>
void
particle_base<ParticleClass>::clear_guard_cells() {
  // erase_ptc_in_guard_cells(m_data.cell, m_number);
}

template <typename ParticleClass>
void
particle_base<ParticleClass>::compute_spectrum(
    int num_bins, std::vector<Scalar>& energies,
    std::vector<uint32_t>& nums) {
  // // Assume the particle energies have been computed
  // energies.resize(num_bins, 0.0);
  // nums.resize(num_bins, 0);

  // // Find maximum energy in the array now
  // thrust::device_ptr<Scalar> E_ptr =
  //     thrust::device_pointer_cast(m_data.E);
  // Scalar E_max = *thrust::max_element(E_ptr, E_ptr + m_number);
  // // Logger::print_info("Maximum energy is {}", E_max);

  // // Partition the energy bin up to max energy times a factor
  // Scalar dlogE = std::log(E_max) / (Scalar)num_bins;
  // for (int i = 0; i < num_bins; i++) {
  //   energies[i] = std::exp((0.5f + (Scalar)i) * dlogE);
  //   // Logger::print_info("{}", energies[i]);
  // }

  // // Do a histogram
  // uint32_t* d_energies;
  // cudaMalloc(&d_energies, num_bins * sizeof(uint32_t));
  // thrust::device_ptr<uint32_t> ptr_energies =
  //     thrust::device_pointer_cast(d_energies);
  // thrust::fill_n(ptr_energies, num_bins, 0);
  // cudaDeviceSynchronize();

  // compute_energy_histogram(d_energies, m_data.E, m_number, num_bins,
  //                          E_max);

  // // Copy the resulting histogram to output
  // cudaMemcpy(nums.data(), d_energies, num_bins * sizeof(uint32_t),
  //            cudaMemcpyDeviceToHost);

  // cudaFree(d_energies);
}

}  // namespace Aperture

#endif  // _PARTICLE_BASE_IMPL_H_
