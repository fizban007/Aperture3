#ifndef _PARTICLE_BASE_IMPL_H_
#define _PARTICLE_BASE_IMPL_H_

#include "boost/container/vector.hpp"
#include "boost/fusion/include/for_each.hpp"
#include "boost/fusion/include/size.hpp"
#include "boost/fusion/include/zip_view.hpp"
#include "data/detail/particle_data_impl.hpp"
#include "data/particle_base.h"
#include "utils/memory.h"
#include "utils/timer.h"
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <type_traits>
// #include "types/particles.h"

namespace Aperture {

using boost::fusion::at_c;

template <typename ParticleClass>
ParticleBase<ParticleClass>::ParticleBase()
    : m_numMax(0), m_number(0), m_sorted(true), m_data_ptr(nullptr) {
  boost::fusion::for_each(m_data, [](auto& x) { x = nullptr; });
}

template <typename ParticleClass>
ParticleBase<ParticleClass>::ParticleBase(std::size_t max_num)
    : m_numMax(max_num), m_number(0), m_sorted(true) {
  std::cout << "New particle array with size " << max_num << std::endl;
  alloc_mem(max_num);
  m_index.resize(max_num, 0);
  m_index_bak.resize(max_num, 0);
  initialize();
}

template <typename ParticleClass>
ParticleBase<ParticleClass>::ParticleBase(
    const ParticleBase<ParticleClass>& other) {
  std::size_t n = other.m_numMax;
  m_numMax = n;
  m_number = other.m_number;
  m_sorted = other.m_sorted;

  alloc_mem(n);
  m_index.resize(n);
  m_index_bak.resize(n);
  copy_from(other, n);
}

template <typename ParticleClass>
ParticleBase<ParticleClass>::ParticleBase(ParticleBase<ParticleClass>&& other) {
  m_numMax = other.m_numMax;
  m_number = other.m_number;
  m_sorted = other.m_sorted;

  // m_data_ptr = other.m_data_ptr;
  m_data = other.m_data;
  m_index.resize(other.m_numMax);
  m_index_bak.resize(other.m_numMax);

  boost::fusion::for_each(other.m_data, [](auto& x) { x = nullptr; });
  // other.m_data_ptr = nullptr;
}

template <typename ParticleClass>
ParticleBase<ParticleClass>::~ParticleBase() {
  free_mem();
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
  m_index.resize(max_num, 0);
  m_index_bak.resize(max_num, 0);
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
      [this, pos, amount](const auto& x) {
        std::fill_n(boost::fusion::at_c<0>(x) + pos, amount,
                    boost::fusion::at_c<1>(x));
      });
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
                          [this, pos](const auto& x) {
                            boost::fusion::at_c<0>(x)[pos] =
                                boost::fusion::at_c<1>(x);
                          });
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
                          [this, pos](const auto& x) {
                            boost::fusion::at_c<0>(x)[pos] =
                                boost::fusion::at_c<1>(x);
                          });
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
      [num, src_pos, dest_pos](const auto& x) {
        std::copy(boost::fusion::at_c<1>(x) + src_pos,
                  boost::fusion::at_c<1>(x) + src_pos + num,
                  boost::fusion::at_c<0>(x) + dest_pos);
      });
  // Adjust the new number of particles in the array
  if (dest_pos + num > m_number) m_number = dest_pos + num;
}

template <typename ParticleClass>
void
ParticleBase<ParticleClass>::rearrange(std::vector<Index_t>& index,
                                       std::size_t num) {
  // std::cout << "In rearrange!" << std::endl;
  ParticleClass p_tmp;
  if (num == 0) num = index.size();
  for (Index_t i = 0; i < num; i++) {
    // -1 means LLONG_MAX for unsigned long int
    if (index[i] != (Index_t)-1) {
      p_tmp = m_data[i];
      for (Index_t j = i;;) {
        if (index[j] != i) {
          // put(index[j], m_data[j]);
          swap(index[j], p_tmp);
          Index_t id = index[j];
          index[j] = (Index_t)-1;  // Mark as done
          j = id;
        } else {
          put(i, p_tmp);
          index[j] = (Index_t)-1;  // Mark as done
          break;
        }
      }
    }
  }
}

template <typename ParticleClass>
template <typename T>
void
ParticleBase<ParticleClass>::rearrange_single_array(T* array,
                                                    std::vector<Index_t>& index,
                                                    std::size_t num) {
  // std::cout << "In rearrange!" << std::endl;
  T tmp, tmp_swap;
  if (num == 0) num = index.size();
  for (Index_t i = 0; i < num; i++) {
    // -1 means LLONG_MAX for unsigned long int
    if (index[i] != (Index_t)-1) {
      tmp = array[i];
      for (Index_t j = i;;) {
        if (index[j] != i) {
          // put(index[j], m_data[j]);
          // swap(index[j], p_tmp);
          tmp_swap = array[index[j]];
          array[index[j]] = tmp;
          tmp = tmp_swap;
          Index_t id = index[j];
          index[j] = (Index_t)-1;  // Mark as done
          j = id;
        } else {
          // put(i, p_tmp);
          array[i] = tmp;
          index[j] = (Index_t)-1;  // Mark as done
          break;
        }
      }
    }
  }
}

template <typename ParticleClass>
void
ParticleBase<ParticleClass>::rearrange_arrays(std::vector<Index_t>& index,
                                              std::size_t num) {
  // typedef boost::fusion::vector<array_type&> seq;
  boost::fusion::for_each(m_data, [this, &index, num](const auto x) {
    // at_c<0>(x)[pos] = at_c<1>(x);
    std::copy(index.begin(), index.begin() + num, this -> m_index_bak.begin());
    this->rearrange_single_array(x, this->m_index_bak, num);
  });
}

template <typename ParticleClass>
void
ParticleBase<ParticleClass>::rearrange_copy(std::vector<Index_t>& index,
                                            std::size_t num) {
  boost::fusion::for_each(m_data, [this, &index, num](const auto x) {
      // at_c<0>(x)[pos] = at_c<1>(x);
      // this->rearrange_single_array(x, index, num);

    });
}

template <typename ParticleClass>
void
ParticleBase<ParticleClass>::partition(std::vector<Index_t>& partitions,
                                       const Grid& grid) {
  // timer::stamp();
  // TODO: This process is tediously slow. How to optimize?

  // std::cout << "In partition!" << std::endl;
  // unsigned int zone_num = 27 + grid.size(); // FIXME: Magic numbers!
  unsigned int zone_num = 27u;  // FIXME: Magic numbers!
  if (partitions.size() != zone_num + 2) partitions.resize(zone_num + 2);

  std::fill(partitions.begin(), partitions.end(), 0);
  std::iota(m_index.begin(), m_index.end(), 0);

  std::cout << "Partitions has size " << partitions.size() << std::endl;
  // std::cout << "Array has size " << m_number << std::endl;
  for (Index_t i = 0; i < m_number; i++) {
    unsigned int zone_idx = 0;
    if (is_empty(i)) {
      zone_idx = zone_num;
    } else {
      zone_idx = grid.mesh().find_zone(m_data.cell[i]);
    }
    // if (zone_idx == CENTER_ZONE) // FIXME: Magic number again!!?
    //   zone_idx = m_data.cell[i];
    // else if (zone_idx != zone_num)
    //   zone_idx += grid.size();
    // Right now m_index array saves the id of each particle in its
    // zone, and partitions array saves the number of particles in
    // each zone
    m_index[i] = partitions[zone_idx + 1];
    partitions[zone_idx + 1] += 1;
  }
  // for (auto n : m_index) { std::cout << n << " "; }
  // std::cout << std::endl;

  // Scan the array, now the array contains the starting index of each
  // zone in the main particle array
  for (unsigned int i = 1; i < zone_num + 2; i++) {
    partitions[i] += partitions[i - 1];
    // The last element means how many particles are empty
  }
  // Second pass through the particle array, get the real index
  for (Index_t i = 0; i < m_number; i++) {
    unsigned int zone_idx = 0;
    if (is_empty(i)) {
      zone_idx = zone_num;
    } else {
      zone_idx = grid.mesh().find_zone(m_data.cell[i]);
    }
    // if (zone_idx == CENTER_ZONE) // FIXME: Magic number again!!?
    //   zone_idx = m_data.cell[i];
    // else if (zone_idx != zone_num)
    //   zone_idx += grid.size();
    m_index[i] += partitions[zone_idx];
  }
  // std::copy(m_index.begin(), m_index.begin() + m_number, m_index_bak.begin());
  // for (auto n : m_index) { std::cout << n << " "; }
  // std::cout << std::endl;

  // Rearrange the particles to reflect the partition
  // timer::show_duration_since_stamp("partition", "ms");
  rearrange_arrays(m_index, m_number);
  // rearrange(m_index, m_number);
  // timer::show_duration_since_stamp("rearrange", "ms");

  // for (int i = 0; i < numMax(); i++) {
  //   std::cout << m_data.cell[i] << " ";
  // }
  // std::cout << std::endl;
  // partitions[zone_num] is where the empty zone starts. This should
  // be equal to the number of particles in the array now
  // FIXME: There could be wrap around error due to large number of particles
  if (partitions[zone_num] != m_number) this->set_num(partitions[zone_num]);

  // std::cout << "Finished partition!" << std::endl;
}

template <typename ParticleClass>
void
ParticleBase<ParticleClass>::partition_and_sort(
    std::vector<Index_t>& partitions, const Aperture::Grid& grid,
    int tile_size) {
  // Make sure the tile size divides the reduced dimension in every direction
  for (int i = 0; i < 3; i++) {
    if (grid.mesh().dims[i] > 1 &&
        grid.mesh().reduced_dim(i) % tile_size != 0) {
      std::cerr << "Tile size does not divide the dimension in direction " << i
                << std::endl;
      return;
    }
  }

  // Compute the number of tiles
  int num_tiles[3] = {1, 1, 1};
  int total_num_tiles = 1;
  for (int i = 0; i < 3; i++) {
    if (grid.mesh().dims[i] > 1)
      num_tiles[i] = grid.mesh().reduced_dim(i) / tile_size;
    total_num_tiles *= num_tiles[i];
  }

  // unsigned int zone_num = 27 + grid.size(); // FIXME: Magic numbers!
  unsigned int zone_num = 27u + total_num_tiles;  // FIXME: Magic numbers!
  if (partitions.size() != zone_num + 2) partitions.resize(zone_num + 2);

  std::fill(partitions.begin(), partitions.end(), 0);
  std::iota(m_index.begin(), m_index.end(), 0);

  std::cout << "Partitions has size " << partitions.size() << std::endl;
  // std::cout << "Array has size " << m_number << std::endl;
  for (Index_t i = 0; i < m_number; i++) {
    unsigned int zone_idx = 0;
    if (is_empty(i)) {
      zone_idx = zone_num;
    } else {
      zone_idx = grid.mesh().find_zone(m_data.cell[i]);
    }
    if (zone_idx == CENTER_ZONE) {
      zone_idx = grid.mesh().tile_id(m_data.cell[i], tile_size);
    } else if (zone_idx != zone_num) {
      zone_idx += total_num_tiles;
    }
    // Right now m_index array saves the id of each particle in its
    // zone, and partitions array saves the number of particles in
    // each zone
    m_index[i] = partitions[zone_idx + 1];
    partitions[zone_idx + 1] += 1;
  }

  // Scan the array, now the array contains the starting index of each
  // zone in the main particle array
  for (unsigned int i = 1; i < zone_num + 2; i++) {
    partitions[i] += partitions[i - 1];
    // The last element means how many particles are empty
  }
  // Second pass through the particle array, get the real index
  for (Index_t i = 0; i < m_number; i++) {
    unsigned int zone_idx = 0;
    if (is_empty(i)) {
      zone_idx = zone_num;
    } else {
      zone_idx = grid.mesh().find_zone(m_data.cell[i]);
    }
    if (zone_idx == CENTER_ZONE) {
      zone_idx = grid.mesh().tile_id(m_data.cell[i], tile_size);
    } else if (zone_idx != zone_num) {
      zone_idx += total_num_tiles;
    }
    m_index[i] += partitions[zone_idx];
  }

  // std::copy(m_index.begin(), m_index.begin() + m_number, m_index_bak.begin());
  // Rearrange the particles to reflect the partition
  // timer::show_duration_since_stamp("partition", "ms");
  rearrange_arrays(m_index, m_number);
  // rearrange(m_index, m_number);
  // timer::show_duration_since_stamp("rearrange", "ms");

  // partitions[zone_num] is where the empty zone starts. This should
  // be equal to the number of particles in the array now
  // FIXME: There could be wrap around error due to large number of particles
  if (partitions[zone_num] != m_number) this->set_num(partitions[zone_num]);
}

template <typename ParticleClass>
void
ParticleBase<ParticleClass>::copy_from(const std::vector<ParticleClass>& buffer,
                                       std::size_t num, std::size_t src_pos,
                                       std::size_t dest_pos) {
  // Adjust the number so that we don't over fill
  if (dest_pos > m_numMax)
    throw std::runtime_error("Destination position larger than buffer size!");
  if (dest_pos + num > m_numMax) num = m_numMax - dest_pos;
  typedef boost::fusion::vector<array_type&, const ParticleClass&> seq;
  for (Index_t i = 0; i < num; i++) {
    boost::fusion::for_each(
        boost::fusion::zip_view<seq>(seq(m_data, buffer[src_pos + i])),
        [this, num, dest_pos, i](const auto& x) {
          // std::copy(at_c<1>(x) + src_pos, at_c<1>(x) + src_pos + num,
          // at_c<0>(x) + dest_pos);
          boost::fusion::at_c<0>(x)[dest_pos + i] = boost::fusion::at_c<1>(x);
        });
  }
  // Adjust the new number of particles in the array
  if (dest_pos + num > m_number) m_number = dest_pos + num;
}

template <typename ParticleClass>
void
ParticleBase<ParticleClass>::copy_to_buffer(std::vector<ParticleClass>& buffer,
                                            std::size_t num,
                                            std::size_t src_pos,
                                            std::size_t dest_pos) {
  // check if we overflow
  if (dest_pos > buffer.size())
    throw std::runtime_error("Destination position larger than buffer size!");
  if (dest_pos + num > buffer.size()) num = buffer.size() - dest_pos;

  typedef boost::fusion::vector<ParticleClass&, const array_type&> seq;
  for (Index_t i = 0; i < num; i++) {
    boost::fusion::for_each(
        boost::fusion::zip_view<seq>(seq(buffer[dest_pos + i], m_data)),
        [this, num, src_pos, i](const auto& x) {
          // std::copy(at_c<1>(x) + src_pos, at_c<1>(x) + src_pos + num,
          // at_c<0>(x) + dest_pos);
          boost::fusion::at_c<0>(x) = boost::fusion::at_c<1>(x)[src_pos + i];
        });
  }
}

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
