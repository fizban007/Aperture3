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
#include <omp.h>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace Aperture {

template <typename ParticleClass>
particle_base<ParticleClass>::particle_base() {
  visit_struct::for_each(
      m_data, [](const char* name, auto& x) { x = nullptr; });
}

template <typename ParticleClass>
particle_base<ParticleClass>::particle_base(std::size_t max_num)
    : particle_interface(max_num) {
  std::cout << "New particle array with size " << max_num << std::endl;
  alloc_mem(max_num);

  initialize();
}

template <typename ParticleClass>
particle_base<ParticleClass>::particle_base(
    const particle_base<ParticleClass>& other) {
  m_size = other.m_size;
  m_number = other.m_number;

  alloc_mem(m_size);
  copy_from(other, m_number);
}

template <typename ParticleClass>
particle_base<ParticleClass>::particle_base(
    particle_base<ParticleClass>&& other) {
  m_size = other.m_size;
  m_number = other.m_number;

  m_data = other.m_data;
  m_index.resize(m_size);
  visit_struct::for_each(
      other.m_data, [](const char* name, auto& x) { x = nullptr; });
}

template <typename ParticleClass>
particle_base<ParticleClass>::~particle_base() {
  free_mem();
}

template <typename ParticleClass>
void
particle_base<ParticleClass>::alloc_mem(std::size_t max_num,
                                        std::size_t alignment) {
  // alloc_struct_of_arrays(m_data, max_num);
  visit_struct::for_each(m_data, [max_num, alignment](const char* name,
                                                      auto& x) {
    typedef typename std::remove_reference<decltype(*x)>::type x_type;
    void* p = aligned_malloc(max_num * sizeof(x_type), alignment);
    x = reinterpret_cast<
        typename std::remove_reference<decltype(x)>::type>(p);
  });

  // Allocate the index array
  m_index.resize(max_num, 0);
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

  initialize();
}

template <typename ParticleClass>
void
particle_base<ParticleClass>::erase(std::size_t pos,
                                    std::size_t amount) {
  if (pos + amount > m_size) amount = m_size - pos;
  std::cout << "Erasing from " << pos << " for " << amount
            << " number of particles" << std::endl;

  if (amount == 1)
    m_data.cell[pos] = MAX_CELL;
  else
    std::fill_n(m_data.cell + pos, amount, MAX_CELL);
}

template <typename ParticleClass>
void
particle_base<ParticleClass>::put(Index_t pos,
                                  const ParticleClass& part) {
  if (pos >= m_size)
    throw std::runtime_error(
        "Trying to insert particle beyond the end of the array. Resize "
        "it first!");

  for_each_arg(m_data, part, [pos](auto& t, auto& u) { t[pos] = u; });
  if (pos >= m_number) m_number = pos + 1;
}

template <typename ParticleClass>
void
particle_base<ParticleClass>::swap(Index_t pos, ParticleClass& part) {
  ParticleClass p_tmp = m_data[pos];
  if (pos >= m_size)
    throw std::runtime_error(
        "Trying to swap particle beyond the end of the array. Resize "
        "it first!");

  for_each_arg(m_data, part, [pos](auto& t, auto& u) { t[pos] = u; });
  part = p_tmp;
  if (pos >= m_number) m_number = pos + 1;
}

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
  if (dest_pos + num > m_size) num = m_size - dest_pos;
  for_each_arg(m_data, other.m_data,
               [num, src_pos, dest_pos](auto& t, auto& u) {
                 std::copy_n(u + src_pos, num, t + dest_pos);
               });
  // Adjust the new number of particles in the array
  if (dest_pos + num > m_number) m_number = dest_pos + num;
}

template <typename ParticleClass>
void
particle_base<ParticleClass>::sort_by_cell(const Grid& grid) {
  // Compute the number of cells and resize the partition array if
  // needed
  uint32_t num_cells = grid.mesh().size();
  if (m_partition.size() != num_cells + 2)
    m_partition.resize(num_cells + 2);

  std::fill(m_partition.begin(), m_partition.end(), 0);
  // Generate particle index from 0 up to the current number
  std::iota(m_index.begin(), m_index.begin() + m_number, 0);

// #pragma omp simd
  // Loop over the particle array to count how many particles in each
  // cell
  for (std::size_t i = 0; i < m_number; i++) {
    Index_t cell_idx = 0;
    if (is_empty(i))
      cell_idx = num_cells;
    else
      cell_idx = m_data.cell[i];
    // Right now m_index array saves the id of each particle in its
    // cell, and partitions array saves the number of particles in
    // each cell
    m_index[i] = m_partition[cell_idx + 1];
    m_partition[cell_idx + 1] += 1;
  }

  // Scan the array, now the array contains the starting index of each
  // zone in the main particle array
  for (uint32_t i = 1; i < num_cells + 2; i++) {
    m_partition[i] += m_partition[i - 1];
    // The last element means how many particles are empty
  }

  // Second pass through the particle array, get the real index
  for (Index_t i = 0; i < m_number; i++) {
    Index_t cell_idx = 0;
    if (is_empty(i)) {
      cell_idx = num_cells;
    } else {
      cell_idx = m_data.cell[i];
    }
    m_index[i] += m_partition[cell_idx];
  }

  // Rearrange the particles to reflect the partition
  // timer::show_duration_since_stamp("partition", "ms");
  rearrange_arrays();

  // num_cells is where the empty particles start, so we record this as
  // the new particle number
  if (m_partition[num_cells] != m_number)
    set_num(m_partition[num_cells]);
}

template <typename ParticleClass>
void
particle_base<ParticleClass>::rearrange_arrays() {
  // const uint32_t padding = 100;
  ParticleClass p_tmp;
  for (Index_t i = 0; i < m_number; i++) {
    // -1 means LLONG_MAX for unsigned long int
    if (m_index[i] != (Index_t)-1) {
      p_tmp = m_data[i];
      for (Index_t j = i;;) {
        if (m_index[j] != i) {
          // put(index[j], m_data[j]);
          swap(m_index[j], p_tmp);
          Index_t id = m_index[j];
          m_index[j] = (Index_t)-1;  // Mark as done
          j = id;
        } else {
          put(i, p_tmp);
          m_index[j] = (Index_t)-1;  // Mark as done
          break;
        }
      }
    }
  }
}

template <typename ParticleClass>
void
particle_base<ParticleClass>::clear_guard_cells() {
  // erase_ptc_in_guard_cells(m_data.cell, m_number);
}

// template <typename ParticleClass>
// void
// particle_base<ParticleClass>::compute_spectrum(
//     int num_bins, std::vector<Scalar>& energies,
//     std::vector<uint32_t>& nums) {
//   // // Assume the particle energies have been computed
//   // energies.resize(num_bins, 0.0);
//   // nums.resize(num_bins, 0);

//   // // Find maximum energy in the array now
//   // thrust::device_ptr<Scalar> E_ptr =
//   //     thrust::device_pointer_cast(m_data.E);
//   // Scalar E_max = *thrust::max_element(E_ptr, E_ptr + m_number);
//   // // Logger::print_info("Maximum energy is {}", E_max);

//   // // Partition the energy bin up to max energy times a factor
//   // Scalar dlogE = std::log(E_max) / (Scalar)num_bins;
//   // for (int i = 0; i < num_bins; i++) {
//   //   energies[i] = std::exp((0.5f + (Scalar)i) * dlogE);
//   //   // Logger::print_info("{}", energies[i]);
//   // }

//   // // Do a histogram
//   // uint32_t* d_energies;
//   // cudaMalloc(&d_energies, num_bins * sizeof(uint32_t));
//   // thrust::device_ptr<uint32_t> ptr_energies =
//   //     thrust::device_pointer_cast(d_energies);
//   // thrust::fill_n(ptr_energies, num_bins, 0);
//   // cudaDeviceSynchronize();

//   // compute_energy_histogram(d_energies, m_data.E, m_number,
//   num_bins,
//   //                          E_max);

//   // // Copy the resulting histogram to output
//   // cudaMemcpy(nums.data(), d_energies, num_bins * sizeof(uint32_t),
//   //            cudaMemcpyDeviceToHost);

//   // cudaFree(d_energies);
// }

}  // namespace Aperture

#endif  // _PARTICLE_BASE_IMPL_H_
