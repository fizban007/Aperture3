#ifndef _PARTICLE_BASE_H_
#define _PARTICLE_BASE_H_

#include "core/enum_types.h"
#include "core/grid.h"
#include "data/particle_data.h"
#include "data/particle_interface.h"
#include <cstddef>
#include <vector>

namespace Aperture {

////////////////////////////////////////////////////////////////////////////////
///  Base class for particle storage on CPU memory. 
////////////////////////////////////////////////////////////////////////////////
template <typename ParticleClass>
class particle_base : public particle_interface {
 protected:
  typedef typename particle_array_type<ParticleClass>::type array_type;

  array_type m_data;
  std::vector<Index_t> m_index;
  std::vector<Index_t> m_partition;

 public:
  /// Default constructor, initializing everything to 0 and set pointers
  /// to nullptr
  particle_base();

  /// Main constructor, initializing the max number to a given value
  /// and current number to zero.
  /// @param max_num Target maximum number.
  explicit particle_base(std::size_t max_num);

  particle_base(const particle_base<ParticleClass>& other);
  particle_base(particle_base<ParticleClass>&& other);

  /// Virtual destructor because we need to derive from this class.
  virtual ~particle_base();

  void resize(std::size_t max_num);
  void initialize();
  void erase(std::size_t pos, std::size_t amount = 1);
  void copy_from(const particle_base<ParticleClass>& other,
                 std::size_t num, std::size_t src_pos = 0,
                 std::size_t dest_pos = 0);
  // void copy_from(const std::vector<ParticleClass>& buffer,
  // std::size_t num, std::size_t src_pos = 0, std::size_t dest_pos =
  // 0); void copy_to_buffer(std::vector<ParticleClass>& buffer,
  // std::size_t num, std::size_t src_pos = 0, std::size_t dest_pos =
  // 0);

  // void put(std::size_t pos, const Vec3<Pos_t>& x, const Vec3<Mom_t>&
  // p, int cell, int flag = 0);
  void put(Index_t pos, const ParticleClass& part);
  void append(const ParticleClass& part);
  void swap(Index_t pos, ParticleClass& part);

  // void compute_tile_num();
  // void sort_by_tile();
  void sort_by_cell(const Grid& grid);
  // void rearrange_arrays(const std::string& skip);

  void clear_guard_cells(const Grid& grid);

  // void compute_spectrum(int num_bins, std::vector<Scalar>& energies,
  //                       std::vector<uint32_t>& nums);

  // void sync_to_device(int deviceId = 0);
  // void sync_to_host();

  // Accessor methods
  array_type& data() { return m_data; }
  const array_type& data() const { return m_data; }

  /// Test if a given position in the particle array is empty
  bool is_empty(Index_t pos) const {
    // Note we need to assume any particle array type has a member
    // called "cell"
    return (m_data.cell[pos] == MAX_CELL);
  }

 protected:
  void alloc_mem(std::size_t max_num, std::size_t alignment = 64);
  void free_mem();

  void rearrange_arrays();
};  // ----- end of class particle_base -----

}  // namespace Aperture

// #include "data/detail/particle_base_impl_dev.hpp"

#endif  // _PARTICLE_BASE_H_
