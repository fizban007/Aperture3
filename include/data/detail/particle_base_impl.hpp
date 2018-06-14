#ifndef _PARTICLE_BASE_IMPL_H_
#define _PARTICLE_BASE_IMPL_H_

#include "boost/container/vector.hpp"
#include "boost/fusion/include/for_each.hpp"
#include "boost/fusion/include/size.hpp"
#include "boost/fusion/include/zip_view.hpp"
#include "boost/mpl/range_c.hpp"
#include "boost/mpl/integral_c.hpp"
#include "data/detail/particle_data_impl.hpp"
#include "data/particle_base.h"
#include "utils/memory.h"
#include "utils/timer.h"
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <algorithm>
#include <numeric>
#include <string>
#include <stdexcept>
#include <type_traits>
// #include "types/particles.h"

namespace Aperture {

namespace Kernels {

// FIXME: This is only for 1D
__global__
void compute_tile(uint32_t* tile, const uint32_t* cell, size_t N, int tile_size) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < N;
       i += blockDim.x * gridDim.x) {
    tile[i] = cell[i] / tile_size;
  }
}

}

// using boost::fusion::at_c;

struct set_nullptr {
  template <typename T>
  HD_INLINE void operator()(T& x) const { x = nullptr; }
};

struct assign_at_idx {
  size_t idx_;
  HOST_DEVICE assign_at_idx(size_t idx) : idx_(idx) {}

  template <typename T>
  HD_INLINE void operator()(const T& x) const {
    boost::fusion::at_c<0>(x)[idx_] = boost::fusion::at_c<1>(x);
  }
};

struct fill_pos_amount {
  size_t pos_, amount_;

  fill_pos_amount(size_t pos, size_t amount) :
      pos_(pos), amount_(amount) {}

  template <typename T>
  void operator()(const T& x) const {
    std::fill_n(boost::fusion::at_c<0>(x) + pos_, amount_,
                boost::fusion::at_c<1>(x));
  }
};

struct sync_dev {
  int devId_;
  size_t size_;

  sync_dev(int devId, size_t size) :
      devId_(devId), size_(size) {}

  template <typename T>
  void operator()(T& x) const {
    typedef typename std::remove_reference<decltype(*x)>::type x_type;
    cudaMemPrefetchAsync(x, size_ * sizeof(x_type), devId_);
  }
};

struct copy_to_dest {
  size_t num_, src_pos_, dest_pos_;
  copy_to_dest(size_t num, size_t src_pos, size_t dest_pos) :
      num_(num), src_pos_(src_pos), dest_pos_(dest_pos) {}

  template <typename T>
  void operator()(const T& x) const {
    std::copy(boost::fusion::at_c<1>(x) + src_pos_,
              boost::fusion::at_c<1>(x) + src_pos_ + num_,
              boost::fusion::at_c<0>(x) + dest_pos_);
  }
};

template <typename ArrayType>
struct rearrange_array {
  ArrayType& array_;
  // thrust::device_ptr<Index_t>& index_;
  Index_t* index_;
  size_t N_;
  void* tmp_ptr_;
  std::string skip_;

  rearrange_array(ArrayType& array, Index_t* index,
                  size_t N, void* tmp_ptr, const std::string& skip) :
      array_(array), index_(index), N_(N), tmp_ptr_(tmp_ptr), skip_(skip) {}

  template <typename T>
  void operator()(T i) const {
    auto ptr_index = thrust::device_pointer_cast(index_);
    auto name = boost::fusion::extension::struct_member_name<ArrayType, i>::call();
    if (name == "cell" || name == "dx1") return;

    // printf("%d, %s\n", T::value, name);
    auto x = boost::fusion::at_c<T::value>(array_);
    auto x_ptr = thrust::device_pointer_cast(x);
    auto tmp_ptr = thrust::device_pointer_cast(reinterpret_cast<decltype(x)>(tmp_ptr_));
    thrust::gather(ptr_index, ptr_index + N_, x_ptr, tmp_ptr);
    thrust::copy_n(tmp_ptr, N_, x_ptr);
  }
};

template <typename ParticleClass>
ParticleBase<ParticleClass>::ParticleBase()
    : m_numMax(0), m_number(0), m_tmp_data_ptr(nullptr), m_index(nullptr) {
  boost::fusion::for_each(m_data, set_nullptr());
}

template <typename ParticleClass>
ParticleBase<ParticleClass>::ParticleBase(std::size_t max_num)
    : m_numMax(max_num), m_number(0) {
  std::cout << "New particle array with size " << max_num << std::endl;
  alloc_mem(max_num);
  auto alloc = alloc_cuda_managed(max_num);
  alloc(m_index);
  cudaMallocManaged(&m_tmp_data_ptr, max_num*sizeof(double));
  // m_index.resize(max_num, 0);
  // m_index_bak.resize(max_num, 0);
  initialize();
}

template <typename ParticleClass>
ParticleBase<ParticleClass>::ParticleBase(
    const ParticleBase<ParticleClass>& other) {
  std::size_t n = other.m_numMax;
  m_numMax = n;
  m_number = other.m_number;
  // m_sorted = other.m_sorted;

  alloc_mem(n);
  auto alloc = alloc_cuda_managed(n);
  alloc(m_index);
  cudaMallocManaged(&m_tmp_data_ptr, n*sizeof(double));
  // alloc((double*)m_tmp_data_ptr);
  // m_index.resize(n);
  // m_index_bak.resize(n);
  copy_from(other, n);
}

template <typename ParticleClass>
ParticleBase<ParticleClass>::ParticleBase(ParticleBase<ParticleClass>&& other) {
  m_numMax = other.m_numMax;
  m_number = other.m_number;
  // m_sorted = other.m_sorted;

  // m_data_ptr = other.m_data_ptr;
  m_data = other.m_data;
  auto alloc = alloc_cuda_managed(m_numMax);
  alloc(m_index);
  cudaMallocManaged(&m_tmp_data_ptr, m_numMax*sizeof(double));
  // alloc((double*)m_tmp_data_ptr);
  // m_index.resize(other.m_numMax);
  // m_index_bak.resize(other.m_numMax);

  boost::fusion::for_each(other.m_data, set_nullptr());
  // other.m_data_ptr = nullptr;
}

template <typename ParticleClass>
ParticleBase<ParticleClass>::~ParticleBase() {
  free_mem();
  free_cuda()(m_index);
  free_cuda()(m_tmp_data_ptr);
}

template <typename ParticleClass>
void
ParticleBase<ParticleClass>::alloc_mem(std::size_t max_num) {
  alloc_struct_of_arrays(m_data, max_num);
}

template <typename ParticleClass>
void
ParticleBase<ParticleClass>::free_mem() {
  free_struct_of_arrays(m_data);
}

template <typename ParticleClass>
void
ParticleBase<ParticleClass>::initialize() {
  erase(0, m_numMax);
  m_number = 0;
}

template <typename ParticleClass>
void
ParticleBase<ParticleClass>::resize(std::size_t max_num) {
  m_numMax = max_num;
  if (m_number > max_num) m_number = max_num;
  free_mem();
  alloc_mem(max_num);

  free_cuda()(m_index);
  free_cuda()(m_tmp_data_ptr);
  auto alloc = alloc_cuda_managed(max_num);
  alloc(m_index);
  cudaMallocManaged(&m_tmp_data_ptr, max_num*sizeof(double));
  // alloc((double*)m_tmp_data_ptr);

  // m_index.resize(max_num, 0);
  // m_index_bak.resize(max_num, 0);
  initialize();
}

template <typename ParticleClass>
void
ParticleBase<ParticleClass>::erase(std::size_t pos, std::size_t amount) {
  if (pos + amount > m_numMax) amount = m_numMax - pos;
  // std::cout << "Erasing from " << pos << " for " << amount << " number of
  // particles" << std::endl;

  typedef boost::fusion::vector<array_type&, const ParticleClass&> seq;
  boost::fusion::for_each(
      boost::fusion::zip_view<seq>(seq(m_data, ParticleClass())),
      fill_pos_amount(pos, amount));
}

template <typename ParticleClass>
void
ParticleBase<ParticleClass>::put(Index_t pos, const ParticleClass& part) {
  if (pos >= m_numMax)
    throw std::runtime_error(
        "Trying to insert particle beyond the end of the array. Resize it "
        "first!");

  typedef boost::fusion::vector<array_type&, const ParticleClass&> seq;
  boost::fusion::for_each(boost::fusion::zip_view<seq>(seq(m_data, part)),
                          assign_at_idx(pos) );
  if (pos >= m_number) m_number = pos + 1;
}

template <typename ParticleClass>
void
ParticleBase<ParticleClass>::swap(Index_t pos, ParticleClass& part) {
  ParticleClass p_tmp = m_data[pos];
  if (pos >= m_numMax)
    throw std::runtime_error(
        "Trying to swap particle beyond the end of the array. Resize it "
        "first!");

  typedef boost::fusion::vector<array_type&, const ParticleClass&> seq;
  boost::fusion::for_each(boost::fusion::zip_view<seq>(seq(m_data, part)),
                          assign_at_idx(pos) );
  part = p_tmp;
  if (pos >= m_number) m_number = pos + 1;
}

template <typename ParticleClass>
void
ParticleBase<ParticleClass>::append(const ParticleClass& part) {
  // put(m_number, x, p, cell, flag);
  put(m_number, part);
}

template <typename ParticleClass>
void
ParticleBase<ParticleClass>::copy_from(const ParticleBase<ParticleClass>& other,
                                       std::size_t num, std::size_t src_pos,
                                       std::size_t dest_pos) {
  // Adjust the number so that we don't over fill
  if (dest_pos + num > m_numMax) num = m_numMax - dest_pos;
  typedef boost::fusion::vector<array_type&, const array_type&> seq;
  boost::fusion::for_each(
      boost::fusion::zip_view<seq>(seq(m_data, other.m_data)),
      copy_to_dest(num, src_pos, dest_pos));
  // Adjust the new number of particles in the array
  if (dest_pos + num > m_number) m_number = dest_pos + num;
}

template <typename ParticleClass>
void
ParticleBase<ParticleClass>::compute_tile_num(int tile_size) {
  Kernels::compute_tile<<<256, 256>>>(m_data.tile, m_data.cell, this->m_number);
  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
}

template <typename ParticleClass>
void
ParticleBase<ParticleClass>::sort_by_tile(int tile_size) {
  // First compute the tile number according to current cell id
  Kernels::compute_tile<<<256, 256>>>(m_data.tile, m_data.cell, this->m_number);

  // Generate particle index array
  auto ptr_tile = thrust::device_pointer_cast(m_data.tile);
  auto ptr_idx = thrust::device_pointer_cast(m_index);
  thrust::counting_iterator<Index_t> iter(0);
  thrust::copy_n(iter, this->m_number, ptr_idx);

  // Sort the index array by key
  thrust::sort_by_key(ptr_tile, ptr_tile + this->m_number, ptr_idx);

  // TODO: Move the rest of particle array using the new index
  rearrange_arrays("tile");

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
}

template <typename ParticleClass>
void
ParticleBase<ParticleClass>::sort_by_cell() {
  // Generate particle index array
  auto ptr_cell = thrust::device_pointer_cast(m_data.cell);
  auto ptr_idx = thrust::device_pointer_cast(m_index);
  thrust::counting_iterator<Index_t> iter(0);
  thrust::copy_n(iter, this->m_number, ptr_idx);

  // Sort the index array by key
  thrust::sort_by_key(ptr_cell, ptr_cell + this->m_number, ptr_idx);

  // TODO: Move the rest of particle array using the new index
  rearrange_arrays("cell");

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
}

template <typename ParticleClass>
void
ParticleBase<ParticleClass>::rearrange_arrays(const std::string& skip) {
  // auto ptr_idx = thrust::device_pointer_cast(m_index);
  boost::fusion::for_each(boost::mpl::range_c<
                          unsigned, 0, boost::fusion::result_of::size<array_type>::value>(),
                          rearrange_array<array_type>(m_data, m_index, m_numMax, m_tmp_data_ptr, skip));
}

template <typename ParticleClass>
void
ParticleBase<ParticleClass>::move_tile() {
  // Assuming compute_tile_num is already called

}

template <typename ParticleClass>
void
ParticleBase<ParticleClass>::sync_to_device(int deviceId) {
  boost::fusion::for_each(m_data, sync_dev(deviceId, m_number));
  cudaDeviceSynchronize();
}

template <typename ParticleClass>
void
ParticleBase<ParticleClass>::sync_to_host() {
  boost::fusion::for_each(m_data, sync_dev(cudaCpuDeviceId, m_number));
  cudaDeviceSynchronize();
}
// template <typename ParticleClass>
// void
// ParticleBase<ParticleClass>::rearrange(std::vector<Index_t>& index,
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
// ParticleBase<ParticleClass>::rearrange_single_array(T* array,
//                                                     std::vector<Index_t>& index,
//                                                     std::size_t num) {
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
// ParticleBase<ParticleClass>::rearrange_arrays(std::vector<Index_t>& index,
//                                               std::size_t num) {
//   // typedef boost::fusion::vector<array_type&> seq;
//   boost::fusion::for_each(m_data, [this, &index, num](const auto x) {
//     // at_c<0>(x)[pos] = at_c<1>(x);
//     std::copy(index.begin(), index.begin() + num, this -> m_index_bak.begin());
//     this->rearrange_single_array(x, this->m_index_bak, num);
//   });
// }

// template <typename ParticleClass>
// void
// ParticleBase<ParticleClass>::rearrange_copy(std::vector<Index_t>& index,
//                                             std::size_t num) {
//   boost::fusion::for_each(m_data, [this, &index, num](const auto x) {
//       // at_c<0>(x)[pos] = at_c<1>(x);
//       // this->rearrange_single_array(x, index, num);

//     });
// }

// template <typename ParticleClass>
// void
// ParticleBase<ParticleClass>::partition(std::vector<Index_t>& partitions,
//                                        const Grid& grid) {
//   // timer::stamp();
//   // TODO: This process is tediously slow. How to optimize?

//   // std::cout << "In partition!" << std::endl;
//   // unsigned int zone_num = 27 + grid.size(); // FIXME: Magic numbers!
//   unsigned int zone_num = 27u;  // FIXME: Magic numbers!
//   if (partitions.size() != zone_num + 2) partitions.resize(zone_num + 2);

//   std::fill(partitions.begin(), partitions.end(), 0);
//   std::iota(m_index.begin(), m_index.end(), 0);

//   std::cout << "Partitions has size " << partitions.size() << std::endl;
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

//   // Scan the array, now the array contains the starting index of each
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
//   // std::copy(m_index.begin(), m_index.begin() + m_number, m_index_bak.begin());
//   // for (auto n : m_index) { std::cout << n << " "; }
//   // std::cout << std::endl;

//   // Rearrange the particles to reflect the partition
//   // timer::show_duration_since_stamp("partition", "ms");
//   rearrange_arrays(m_index, m_number);
//   // rearrange(m_index, m_number);
//   // timer::show_duration_since_stamp("rearrange", "ms");

//   // for (int i = 0; i < numMax(); i++) {
//   //   std::cout << m_data.cell[i] << " ";
//   // }
//   // std::cout << std::endl;
//   // partitions[zone_num] is where the empty zone starts. This should
//   // be equal to the number of particles in the array now
//   // FIXME: There could be wrap around error due to large number of particles
//   if (partitions[zone_num] != m_number) this->set_num(partitions[zone_num]);

//   // std::cout << "Finished partition!" << std::endl;
// }

// template <typename ParticleClass>
// void
// ParticleBase<ParticleClass>::partition_and_sort(
//     std::vector<Index_t>& partitions, const Aperture::Grid& grid,
//     int tile_size) {
//   // Make sure the tile size divides the reduced dimension in every direction
//   for (int i = 0; i < 3; i++) {
//     if (grid.mesh().dims[i] > 1 &&
//         grid.mesh().reduced_dim(i) % tile_size != 0) {
//       std::cerr << "Tile size does not divide the dimension in direction " << i
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

//   // unsigned int zone_num = 27 + grid.size(); // FIXME: Magic numbers!
//   unsigned int zone_num = 27u + total_num_tiles;  // FIXME: Magic numbers!
//   if (partitions.size() != zone_num + 2) partitions.resize(zone_num + 2);

//   std::fill(partitions.begin(), partitions.end(), 0);
//   std::iota(m_index.begin(), m_index.end(), 0);

//   std::cout << "Partitions has size " << partitions.size() << std::endl;
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

//   // Scan the array, now the array contains the starting index of each
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

//   // std::copy(m_index.begin(), m_index.begin() + m_number, m_index_bak.begin());
//   // Rearrange the particles to reflect the partition
//   // timer::show_duration_since_stamp("partition", "ms");
//   rearrange_arrays(m_index, m_number);
//   // rearrange(m_index, m_number);
//   // timer::show_duration_since_stamp("rearrange", "ms");

//   // partitions[zone_num] is where the empty zone starts. This should
//   // be equal to the number of particles in the array now
//   // FIXME: There could be wrap around error due to large number of particles
//   if (partitions[zone_num] != m_number) this->set_num(partitions[zone_num]);
// }

// template <typename ParticleClass>
// void
// ParticleBase<ParticleClass>::copy_from(const std::vector<ParticleClass>& buffer,
//                                        std::size_t num, std::size_t src_pos,
//                                        std::size_t dest_pos) {
//   // Adjust the number so that we don't over fill
//   if (dest_pos > m_numMax)
//     throw std::runtime_error("Destination position larger than buffer size!");
//   if (dest_pos + num > m_numMax) num = m_numMax - dest_pos;
//   typedef boost::fusion::vector<array_type&, const ParticleClass&> seq;
//   for (Index_t i = 0; i < num; i++) {
//     boost::fusion::for_each(
//         boost::fusion::zip_view<seq>(seq(m_data, buffer[src_pos + i])),
//         [this, num, dest_pos, i](const auto& x) {
//           // std::copy(at_c<1>(x) + src_pos, at_c<1>(x) + src_pos + num,
//           // at_c<0>(x) + dest_pos);
//           boost::fusion::at_c<0>(x)[dest_pos + i] = boost::fusion::at_c<1>(x);
//         });
//   }
//   // Adjust the new number of particles in the array
//   if (dest_pos + num > m_number) m_number = dest_pos + num;
// }

// template <typename ParticleClass>
// void
// ParticleBase<ParticleClass>::copy_to_buffer(std::vector<ParticleClass>& buffer,
//                                             std::size_t num,
//                                             std::size_t src_pos,
//                                             std::size_t dest_pos) {
//   // check if we overflow
//   if (dest_pos > buffer.size())
//     throw std::runtime_error("Destination position larger than buffer size!");
//   if (dest_pos + num > buffer.size()) num = buffer.size() - dest_pos;

//   typedef boost::fusion::vector<ParticleClass&, const array_type&> seq;
//   for (Index_t i = 0; i < num; i++) {
//     boost::fusion::for_each(
//         boost::fusion::zip_view<seq>(seq(buffer[dest_pos + i], m_data)),
//         [this, num, src_pos, i](const auto& x) {
//           // std::copy(at_c<1>(x) + src_pos, at_c<1>(x) + src_pos + num,
//           // at_c<0>(x) + dest_pos);
//           boost::fusion::at_c<0>(x) = boost::fusion::at_c<1>(x)[src_pos + i];
//         });
//   }
// }

template <typename ParticleClass>
void
ParticleBase<ParticleClass>::clear_guard_cells(const Grid& grid) {
  for (Index_t i = 0; i < m_number; i++) {
    if (!grid.mesh().is_in_bulk(m_data.cell[i])) {
      erase(i);
    }
  }
}
}

#endif  // _PARTICLE_BASE_IMPL_H_
