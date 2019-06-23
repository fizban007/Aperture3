#ifndef _PARTICLE_BASE_H_
#define _PARTICLE_BASE_H_

#include "core/enum_types.h"
#include "core/grid.h"
#include "data/particle_data.h"

namespace Aperture {

///  Base class for particle storage
template <typename ParticleClass>
class particle_base {
 protected:
  typedef typename particle_array_type<ParticleClass>::type array_type;

  array_type m_data;
  array_type m_tracked;
  size_t* m_index = nullptr;
  void* m_tmp_data_ptr = nullptr;
  std::vector<size_t> m_partition;

  size_t m_size = 0;
  size_t m_number = 0;
  uint32_t m_num_tracked = 0;

 public:
  /// Default constructor, initializing everything to 0 and set pointers
  /// to nullptr
  particle_base();

  /// Main constructor, initializing the max number to a given value
  /// and current number to zero.
  /// @param max_num Target maximum number.
  explicit particle_base(std::size_t max_num);

  /// Copy constructor
  particle_base(const particle_base<ParticleClass>& other);

  /// Move Constructor
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

  void put(size_t pos, const ParticleClass& part);
  void append(const ParticleClass& part);
  void swap(size_t pos, ParticleClass& part);

  void sort_by_cell(const Grid& grid);
  // void rearrange_arrays(const std::string& skip);
  void get_tracked_ptc();

  void clear_guard_cells(const Grid& grid);

  // Accessor methods
  array_type& data() { return m_data; }
  const array_type& data() const { return m_data; }
  array_type& tracked_data() { return m_tracked; }
  const array_type& tracked_data() const { return m_tracked; }

  /// @return Returns the value of the current number of particles
  std::size_t number() const { return m_number; }

  /// @return Returns the maximum number of particles
  std::size_t size() const { return m_size; }

  /// @return Returns the number of tracked particles
  uint32_t tracked_number() const { return m_num_tracked; }
 
  /// Set the current number of particles in the array to a given value
  /// @param num New number of particles in the array
  void set_num(size_t num) { m_number = num; }

  /// Test if a given position in the particle array is empty
  bool is_empty(size_t pos) const {
    // Note we need to assume any particle array type has a member
    // called "cell"
    return (m_data.cell[pos] == MAX_CELL);
  }

 protected:
  void alloc_mem(std::size_t max_num, std::size_t alignment = 64);
  void free_mem();

  void rearrange_arrays(const std::string& skip);
};  // ----- end of class particle_base -----


}

#endif  // _PARTICLE_BASE_H_
