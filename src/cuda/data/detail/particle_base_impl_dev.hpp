#ifndef _PARTICLE_BASE_IMPL_DEV_H_
#define _PARTICLE_BASE_IMPL_DEV_H_

#include "cuda/constant_mem.h"
#include "cuda/cudaUtility.h"
#include "cuda/data/particle_base_dev.h"
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

// These are helper functors for for_each loops

struct set_nullptr {
  template <typename T>
  HD_INLINE void operator()(const char* name, T& x) const {
    x = nullptr;
  }
};

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
particle_base_dev<ParticleClass>::particle_base_dev()
    : m_tmp_data_ptr(nullptr), m_index(nullptr) {
  // boost::fusion::for_each(m_data, set_nullptr());
  visit_struct::for_each(m_data, set_nullptr{});
}

template <typename ParticleClass>
particle_base_dev<ParticleClass>::particle_base_dev(std::size_t max_num)
    : particle_interface(max_num) {
  std::cout << "New particle array with size " << max_num << std::endl;
  // CudaSafeCall(cudaGetDevice(&m_devId));
  m_devId = 0;
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
particle_base_dev<ParticleClass>::particle_base_dev(
    const particle_base_dev<ParticleClass>& other) {
  std::size_t n = other.m_size;
  m_size = n;
  m_number = other.m_number;
  m_devId = other.m_devId;
  // CudaSafeCall(cudaSetDevice(m_devId));
  // m_sorted = other.m_sorted;

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
particle_base_dev<ParticleClass>::particle_base_dev(
    particle_base_dev<ParticleClass>&& other) {
  m_size = other.m_size;
  m_number = other.m_number;
  m_devId = other.m_devId;
  // CudaSafeCall(cudaSetDevice(m_devId));
  // m_sorted = other.m_sorted;

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
  visit_struct::for_each(other.m_data, set_nullptr{});
  // other.m_data_ptr = nullptr;
}

template <typename ParticleClass>
particle_base_dev<ParticleClass>::~particle_base_dev() {
  // CudaSafeCall(cudaSetDevice(m_devId));
  free_mem();
  free_cuda()(m_index);
  free_cuda()(m_tmp_data_ptr);
  // CudaSafeCall(cudaFree(m_index));
  // CudaSafeCall(cudaFree(m_tmp_data_ptr));
}

template <typename ParticleClass>
void
particle_base_dev<ParticleClass>::alloc_mem(std::size_t max_num) {
  alloc_struct_of_arrays(m_data, max_num);
}

template <typename ParticleClass>
void
particle_base_dev<ParticleClass>::free_mem() {
  free_struct_of_arrays(m_data);
}

template <typename ParticleClass>
void
particle_base_dev<ParticleClass>::initialize() {
  erase(0, m_size);
  m_number = 0;
}

template <typename ParticleClass>
void
particle_base_dev<ParticleClass>::resize(std::size_t max_num) {
  // CudaSafeCall(cudaSetDevice(m_devId));
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
particle_base_dev<ParticleClass>::erase(std::size_t pos,
                                        std::size_t amount) {
  // CudaSafeCall(cudaSetDevice(m_devId));
  if (pos + amount > m_size) amount = m_size - pos;
  std::cout << "Erasing from index " << pos << " for " << amount
            << " number of particles" << std::endl;

  auto ptc = ParticleClass{};
  for_each_arg(m_data, ptc, fill_pos_amount{pos, amount});
}

template <typename ParticleClass>
void
particle_base_dev<ParticleClass>::put(Index_t pos,
                                      const ParticleClass& part) {
  if (pos >= m_size)
    throw std::runtime_error(
        "Trying to insert particle beyond the end of the array."
        "Resize it first!");

  for_each_arg(m_data, part, assign_at_idx(pos));
  if (pos >= m_number) m_number = pos + 1;
}

// template <typename ParticleClass>
// void
// particle_base_dev<ParticleClass>::swap(Index_t pos, ParticleClass&
// part)
// {
//   ParticleClass p_tmp = m_data[pos];
//   if (pos >= m_size)
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

template <typename ParticleClass>
void
particle_base_dev<ParticleClass>::append(const ParticleClass& part) {
  // put(m_number, x, p, cell, flag);
  put(m_number, part);
}

template <typename ParticleClass>
void
particle_base_dev<ParticleClass>::copy_from(
    const particle_base_dev<ParticleClass>& other, std::size_t num,
    std::size_t src_pos, std::size_t dest_pos) {
  // CudaSafeCall(cudaSetDevice(m_devId));
  // Adjust the number so that we don't over fill
  if (dest_pos + num > m_size) num = m_size - dest_pos;
  for_each_arg(m_data, other.m_data,
               copy_to_dest(num, src_pos, dest_pos));
  // Adjust the new number of particles in the array
  if (dest_pos + num > m_number) m_number = dest_pos + num;
}

// template <typename ParticleClass>
// void
// particle_base_dev<ParticleClass>::compute_tile_num() {
//   compute_tile(m_data.tile, m_data.cell, m_number);
// }

// template <typename ParticleClass>
// void
// particle_base_dev<ParticleClass>::sort_by_tile() {
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
particle_base_dev<ParticleClass>::sort_by_cell() {
  if (m_number > 0) {
    // CudaSafeCall(cudaSetDevice(m_devId));
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
particle_base_dev<ParticleClass>::rearrange_arrays(
    const std::string& skip) {
  const uint32_t padding = 100;
  auto ptc = ParticleClass();
  for_each_arg_with_name(
      m_data, ptc,
      rearrange_array{m_index, std::min(m_size, m_number + padding),
                      m_tmp_data_ptr, skip});
}

// template <typename ParticleClass>
// void
// particle_base_dev<ParticleClass>::copy_to_device(int deviceId) {
//   // boost::fusion::for_each(m_data, sync_dev(deviceId, m_number));
//   visit_struct::for_each(m_data, sync_dev(deviceId, m_number));
//   cudaDeviceSynchronize();
// }

// template <typename ParticleClass>
// void
// particle_base_dev<ParticleClass>::copy_to_host() {
//   // boost::fusion::for_each(m_data, sync_dev(cudaCpuDeviceId,
//   // m_number));
//   visit_struct::for_each(m_data, sync_dev(cudaCpuDeviceId,
//   m_number)); cudaDeviceSynchronize();
// }
// template <typename ParticleClass>
// void
// particle_base_dev<ParticleClass>::rearrange(std::vector<Index_t>&
// index,
//                                        std::size_t num) {
//   // std::cout << "In rearrange!" << std::endl;
//   ParticleClass p_tmp;
//   if (num == 0) num = index.size();
//   for (Index_t i = 0; i < num; i++) {
//     // -1 means LLONG_MAX for unsigned long int
//     if (index[i] != (Index_t)-1) {
//       p_tmp = m_data[i];
//       for (Index_t j = i;;) {
//         if (index[j] != i) {
//           // put(index[j], m_data[j]);
//           swap(index[j], p_tmp);
//           Index_t id = index[j];
//           index[j] = (Index_t)-1;  // Mark as done
//           j = id;
//         } else {
//           put(i, p_tmp);
//           index[j] = (Index_t)-1;  // Mark as done
//           break;
//         }
//       }
//     }
//   }
// }

// template <typename ParticleClass>
// template <typename T>
// void
// particle_base_dev<ParticleClass>::rearrange_single_array(T* array,
//                                                     std::vector<Index_t>&
//                                                     index,
//                                                     std::size_t num)
//                                                     {
//   // std::cout << "In rearrange!" << std::endl;
//   T tmp, tmp_swap;
//   if (num == 0) num = index.size();
//   for (Index_t i = 0; i < num; i++) {
//     // -1 means LLONG_MAX for unsigned long int
//     if (index[i] != (Index_t)-1) {
//       tmp = array[i];
//       for (Index_t j = i;;) {
//         if (index[j] != i) {
//           // put(index[j], m_data[j]);
//           // swap(index[j], p_tmp);
//           tmp_swap = array[index[j]];
//           array[index[j]] = tmp;
//           tmp = tmp_swap;
//           Index_t id = index[j];
//           index[j] = (Index_t)-1;  // Mark as done
//           j = id;
//         } else {
//           // put(i, p_tmp);
//           array[i] = tmp;
//           index[j] = (Index_t)-1;  // Mark as done
//           break;
//         }
//       }
//     }
//   }
// }

// template <typename ParticleClass>
// void
// particle_base_dev<ParticleClass>::rearrange_arrays(std::vector<Index_t>&
// index,
//                                               std::size_t num) {
//   // typedef boost::fusion::vector<array_type&> seq;
//   boost::fusion::for_each(m_data, [this, &index, num](const auto x) {
//     // at_c<0>(x)[pos] = at_c<1>(x);
//     std::copy(index.begin(), index.begin() + num, this ->
//     m_index_bak.begin()); this->rearrange_single_array(x,
//     this->m_index_bak, num);
//   });
// }

// template <typename ParticleClass>
// void
// particle_base_dev<ParticleClass>::rearrange_copy(std::vector<Index_t>&
// index,
//                                             std::size_t num) {
//   boost::fusion::for_each(m_data, [this, &index, num](const auto x) {
//       // at_c<0>(x)[pos] = at_c<1>(x);
//       // this->rearrange_single_array(x, index, num);

//     });
// }

// template <typename ParticleClass>
// void
// particle_base_dev<ParticleClass>::partition(std::vector<Index_t>&
// partitions,
//                                        const Grid& grid) {
//   // timer::stamp();
//   // TODO: This process is tediously slow. How to optimize?

//   // std::cout << "In partition!" << std::endl;
//   // unsigned int zone_num = 27 + grid.size(); // FIXME: Magic
//   numbers! unsigned int zone_num = 27u;  // FIXME: Magic numbers! if
//   (partitions.size() != zone_num + 2) partitions.resize(zone_num +
//   2);

//   std::fill(partitions.begin(), partitions.end(), 0);
//   std::iota(m_index.begin(), m_index.end(), 0);

//   std::cout << "Partitions has size " << partitions.size() <<
//   std::endl;
//   // std::cout << "Array has size " << m_number << std::endl;
//   for (Index_t i = 0; i < m_number; i++) {
//     unsigned int zone_idx = 0;
//     if (is_empty(i)) {
//       zone_idx = zone_num;
//     } else {
//       zone_idx = grid.mesh().find_zone(m_data.cell[i]);
//     }
//     // if (zone_idx == CENTER_ZONE) // FIXME: Magic number again!!?
//     //   zone_idx = m_data.cell[i];
//     // else if (zone_idx != zone_num)
//     //   zone_idx += grid.size();
//     // Right now m_index array saves the id of each particle in its
//     // zone, and partitions array saves the number of particles in
//     // each zone
//     m_index[i] = partitions[zone_idx + 1];
//     partitions[zone_idx + 1] += 1;
//   }
//   // for (auto n : m_index) { std::cout << n << " "; }
//   // std::cout << std::endl;

//   // Scan the array, now the array contains the starting index of
//   each
//   // zone in the main particle array
//   for (unsigned int i = 1; i < zone_num + 2; i++) {
//     partitions[i] += partitions[i - 1];
//     // The last element means how many particles are empty
//   }
//   // Second pass through the particle array, get the real index
//   for (Index_t i = 0; i < m_number; i++) {
//     unsigned int zone_idx = 0;
//     if (is_empty(i)) {
//       zone_idx = zone_num;
//     } else {
//       zone_idx = grid.mesh().find_zone(m_data.cell[i]);
//     }
//     // if (zone_idx == CENTER_ZONE) // FIXME: Magic number again!!?
//     //   zone_idx = m_data.cell[i];
//     // else if (zone_idx != zone_num)
//     //   zone_idx += grid.size();
//     m_index[i] += partitions[zone_idx];
//   }
//   // std::copy(m_index.begin(), m_index.begin() + m_number,
//   m_index_bak.begin());
//   // for (auto n : m_index) { std::cout << n << " "; }
//   // std::cout << std::endl;

//   // Rearrange the particles to reflect the partition
//   // timer::show_duration_since_stamp("partition", "ms");
//   rearrange_arrays(m_index, m_number);
//   // rearrange(m_index, m_number);
//   // timer::show_duration_since_stamp("rearrange", "ms");

//   // for (int i = 0; i < size(); i++) {
//   //   std::cout << m_data.cell[i] << " ";
//   // }
//   // std::cout << std::endl;
//   // partitions[zone_num] is where the empty zone starts. This should
//   // be equal to the number of particles in the array now
//   // FIXME: There could be wrap around error due to large number of
//   particles if (partitions[zone_num] != m_number)
//   this->set_num(partitions[zone_num]);

//   // std::cout << "Finished partition!" << std::endl;
// }

// template <typename ParticleClass>
// void
// particle_base_dev<ParticleClass>::partition_and_sort(
//     std::vector<Index_t>& partitions, const Aperture::Grid& grid,
//     int tile_size) {
//   // Make sure the tile size divides the reduced dimension in every
//   direction for (int i = 0; i < 3; i++) {
//     if (grid.mesh().dims[i] > 1 &&
//         grid.mesh().reduced_dim(i) % tile_size != 0) {
//       std::cerr << "Tile size does not divide the dimension in
//       direction " << i
//                 << std::endl;
//       return;
//     }
//   }

//   // Compute the number of tiles
//   int num_tiles[3] = {1, 1, 1};
//   int total_num_tiles = 1;
//   for (int i = 0; i < 3; i++) {
//     if (grid.mesh().dims[i] > 1)
//       num_tiles[i] = grid.mesh().reduced_dim(i) / tile_size;
//     total_num_tiles *= num_tiles[i];
//   }

//   // unsigned int zone_num = 27 + grid.size(); // FIXME: Magic
//   numbers! unsigned int zone_num = 27u + total_num_tiles;  // FIXME:
//   Magic numbers! if (partitions.size() != zone_num + 2)
//   partitions.resize(zone_num + 2);

//   std::fill(partitions.begin(), partitions.end(), 0);
//   std::iota(m_index.begin(), m_index.end(), 0);

//   std::cout << "Partitions has size " << partitions.size() <<
//   std::endl;
//   // std::cout << "Array has size " << m_number << std::endl;
//   for (Index_t i = 0; i < m_number; i++) {
//     unsigned int zone_idx = 0;
//     if (is_empty(i)) {
//       zone_idx = zone_num;
//     } else {
//       zone_idx = grid.mesh().find_zone(m_data.cell[i]);
//     }
//     if (zone_idx == CENTER_ZONE) {
//       zone_idx = grid.mesh().tile_id(m_data.cell[i], tile_size);
//     } else if (zone_idx != zone_num) {
//       zone_idx += total_num_tiles;
//     }
//     // Right now m_index array saves the id of each particle in its
//     // zone, and partitions array saves the number of particles in
//     // each zone
//     m_index[i] = partitions[zone_idx + 1];
//     partitions[zone_idx + 1] += 1;
//   }

//   // Scan the array, now the array contains the starting index of
//   each
//   // zone in the main particle array
//   for (unsigned int i = 1; i < zone_num + 2; i++) {
//     partitions[i] += partitions[i - 1];
//     // The last element means how many particles are empty
//   }
//   // Second pass through the particle array, get the real index
//   for (Index_t i = 0; i < m_number; i++) {
//     unsigned int zone_idx = 0;
//     if (is_empty(i)) {
//       zone_idx = zone_num;
//     } else {
//       zone_idx = grid.mesh().find_zone(m_data.cell[i]);
//     }
//     if (zone_idx == CENTER_ZONE) {
//       zone_idx = grid.mesh().tile_id(m_data.cell[i], tile_size);
//     } else if (zone_idx != zone_num) {
//       zone_idx += total_num_tiles;
//     }
//     m_index[i] += partitions[zone_idx];
//   }

//   // std::copy(m_index.begin(), m_index.begin() + m_number,
//   m_index_bak.begin());
//   // Rearrange the particles to reflect the partition
//   // timer::show_duration_since_stamp("partition", "ms");
//   rearrange_arrays(m_index, m_number);
//   // rearrange(m_index, m_number);
//   // timer::show_duration_since_stamp("rearrange", "ms");

//   // partitions[zone_num] is where the empty zone starts. This should
//   // be equal to the number of particles in the array now
//   // FIXME: There could be wrap around error due to large number of
//   particles if (partitions[zone_num] != m_number)
//   this->set_num(partitions[zone_num]);
// }

// template <typename ParticleClass>
// void
// particle_base_dev<ParticleClass>::copy_from(const
// std::vector<ParticleClass>& buffer,
//                                        std::size_t num, std::size_t
//                                        src_pos, std::size_t dest_pos)
//                                        {
//   // Adjust the number so that we don't over fill
//   if (dest_pos > m_size)
//     throw std::runtime_error("Destination position larger than buffer
//     size!");
//   if (dest_pos + num > m_size) num = m_size - dest_pos;
//   typedef boost::fusion::vector<array_type&, const ParticleClass&>
//   seq; for (Index_t i = 0; i < num; i++) {
//     boost::fusion::for_each(
//         boost::fusion::zip_view<seq>(seq(m_data, buffer[src_pos +
//         i])), [this, num, dest_pos, i](const auto& x) {
//           // std::copy(at_c<1>(x) + src_pos, at_c<1>(x) + src_pos +
//           num,
//           // at_c<0>(x) + dest_pos);
//           boost::fusion::at_c<0>(x)[dest_pos + i] =
//           boost::fusion::at_c<1>(x);
//         });
//   }
//   // Adjust the new number of particles in the array
//   if (dest_pos + num > m_number) m_number = dest_pos + num;
// }

// template <typename ParticleClass>
// void
// particle_base_dev<ParticleClass>::copy_to_buffer(std::vector<ParticleClass>&
// buffer,
//                                             std::size_t num,
//                                             std::size_t src_pos,
//                                             std::size_t dest_pos) {
//   // check if we overflow
//   if (dest_pos > buffer.size())
//     throw std::runtime_error("Destination position larger than buffer
//     size!");
//   if (dest_pos + num > buffer.size()) num = buffer.size() - dest_pos;

//   typedef boost::fusion::vector<ParticleClass&, const array_type&>
//   seq; for (Index_t i = 0; i < num; i++) {
//     boost::fusion::for_each(
//         boost::fusion::zip_view<seq>(seq(buffer[dest_pos + i],
//         m_data)), [this, num, src_pos, i](const auto& x) {
//           // std::copy(at_c<1>(x) + src_pos, at_c<1>(x) + src_pos +
//           num,
//           // at_c<0>(x) + dest_pos);
//           boost::fusion::at_c<0>(x) =
//           boost::fusion::at_c<1>(x)[src_pos + i];
//         });
//   }
// }

template <typename ParticleClass>
void
particle_base_dev<ParticleClass>::clear_guard_cells() {
  // CudaSafeCall(cudaSetDevice(m_devId));
  erase_ptc_in_guard_cells(m_data.cell, m_number);
}

template <typename ParticleClass>
void
particle_base_dev<ParticleClass>::compute_spectrum(
    int num_bins, std::vector<Scalar>& energies,
    std::vector<uint32_t>& nums) {
  // Assume the particle energies have been computed
  energies.resize(num_bins, 0.0);
  nums.resize(num_bins, 0);

  // Find maximum energy in the array now
  thrust::device_ptr<Scalar> E_ptr =
      thrust::device_pointer_cast(m_data.E);
  Scalar E_max = *thrust::max_element(E_ptr, E_ptr + m_number);
  // Logger::print_info("Maximum energy is {}", E_max);

  // Partition the energy bin up to max energy times a factor
  Scalar dlogE = std::log(E_max) / (Scalar)num_bins;
  for (int i = 0; i < num_bins; i++) {
    energies[i] = std::exp((0.5f + (Scalar)i) * dlogE);
    // Logger::print_info("{}", energies[i]);
  }

  // Do a histogram
  uint32_t* d_energies;
  cudaMalloc(&d_energies, num_bins * sizeof(uint32_t));
  thrust::device_ptr<uint32_t> ptr_energies =
      thrust::device_pointer_cast(d_energies);
  thrust::fill_n(ptr_energies, num_bins, 0);
  // cudaDeviceSynchronize();

  compute_energy_histogram(d_energies, m_data.E, m_number, num_bins,
                           E_max);

  // Copy the resulting histogram to output
  cudaMemcpy(nums.data(), d_energies, num_bins * sizeof(uint32_t),
             cudaMemcpyDeviceToHost);

  cudaFree(d_energies);
}

}  // namespace Aperture

#endif  // _PARTICLE_BASE_IMPL_DEV_H_
