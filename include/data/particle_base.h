#ifndef  _PARTICLE_BASE_H_
#define  _PARTICLE_BASE_H_

#include <cstddef>
#include <vector>
#include "data/grid.h"
#include "data/particle_data.h"
#include "data/enum_types.h"

namespace Aperture {

///  Base class for particle storage classes. The only thing this
///  class does is to maintain a maximum number of particles in the
///  storage buffer, as well as the current particle number that is
///  filled (non-empty). It is the duty of the derived class to
///  specify what to store in the particle array and how to store
///  them. Both the CPU particle buffer and the GPU particle buffer
///  derive from this class.
template <typename ParticleClass>
class ParticleBase
{
 protected:
  typedef typename particle_array_type<ParticleClass>::type array_type;

  std::size_t m_numMax;                  ///< Maximum number of particles in the array
  /// @brief The current number of particles in the array.
  /// @accessors #number(), #setNum()
  std::size_t m_number;

  void* m_tmp_data_ptr;
  array_type m_data;
  Index_t* m_index;
  // std::vector<Index_t> m_index, m_index_bak;
  // bool m_sorted;

 public:
  /// Default constructor, initializing everything to 0 and `sorted` to `true`
  // ParticleBase() : m_numMax(0), m_number(0), m_sorted(true) {}
  ParticleBase();

  /// Main constructor, initializing the max number to a given value
  /// and current number to zero.
  ///
  /// @param max_num Target maximum number.
  ///
  // ParticleBase(size_t max_num) : m_numMax(max_num), m_number(0), m_sorted(true) {}
  explicit ParticleBase(std::size_t max_num);

  ParticleBase(const ParticleBase<ParticleClass>& other);
  ParticleBase(ParticleBase<ParticleClass>&& other);

  /// Virtual destructor because we need to derive from this class.
  virtual ~ParticleBase();

  void alloc_mem(std::size_t max_num);
  void free_mem();

  void resize(std::size_t max_num);
  void initialize();
  void erase(std::size_t pos, std::size_t amount = 1);
  void copy_from(const ParticleBase<ParticleClass>& other, std::size_t num, std::size_t src_pos = 0, std::size_t dest_pos = 0);
  // void copy_from(const std::vector<ParticleClass>& buffer, std::size_t num, std::size_t src_pos = 0, std::size_t dest_pos = 0);
  // void copy_to_buffer(std::vector<ParticleClass>& buffer, std::size_t num, std::size_t src_pos = 0, std::size_t dest_pos = 0);

  // void put(std::size_t pos, const Vec3<Pos_t>& x, const Vec3<Mom_t>& p, int cell, int flag = 0);
  void put(Index_t pos, const ParticleClass& part);
  void append(const ParticleClass& part);
  void swap(Index_t pos, ParticleClass& part);

  void compute_tile_num();
  void sort_by_tile();
  void sort_by_cell();
  void rearrange_arrays(const std::string& skip);
  void move_tile();
  // After rearrange, the index array will all be -1
  // void rearrange(std::vector<Index_t>& index, std::size_t num = 0);
  // void rearrange_arrays(std::vector<Index_t>& index, std::size_t num = 0);
  // void rearrange_copy(std::vector<Index_t>& index, std::size_t num = 0);
  // template <typename T>
  // void rearrange_single_array(T* array, std::vector<Index_t>& index, std::size_t num = 0);
  // Partition according to a grid configuration, sort the particles into the
  // bulk part, and those that needs to be communicated out. After partition,
  // the given array partition becomes the starting position of each partition
  // in the particle array
  // void partition(std::vector<Index_t>& partitions, const Grid& grid);
  // Partition for communication, as well as sorting the particles into tiles,
  // with a given tile size. Tiles have the same size in every direction available
  // void partition_and_sort(std::vector<Index_t>& partitions, const Grid& grid, int tile_size);
  void clear_guard_cells();

  void sync_to_device(int deviceId = 0);
  void sync_to_host();

  // Accessor methods
  array_type& data() { return m_data; }
  const array_type& data() const { return m_data; }
  void* tmp_data() { return m_tmp_data_ptr; };
  const void* tmp_data() const { return m_tmp_data_ptr; };

  bool is_empty(Index_t pos) const {
    // Note we need to assume any particle array type has a member called "cell"
    return (m_data.cell[pos] == MAX_CELL);
  }
  /// @return Returns the value of the current number of particles
  std::size_t number() const { return m_number; }

  /// @return Returns the maximum number of particles
  std::size_t numMax() const { return m_numMax; }

  /// Set the current number of particles in the array to a given value
  ///
  /// @param num New number of particles in the array
  ///
  void set_num(size_t num) { m_number = num; }

  /// Set the array to be sorted.
  // void sorted() { m_sorted = true; }

  /// Set `sorted` to be false.
  // void scrambled() { m_sorted = false; }

}; // ----- end of class ParticleBase -----

}

// #include "data/detail/particle_base_impl.hpp"

#endif   // _PARTICLE_BASE_H_
