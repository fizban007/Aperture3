#ifndef  _PARTICLES_H_
#define  _PARTICLES_H_

#include <cstdlib>
#include <vector>
#include <string>
#include "data/particle_base.h"
#include "constant_defs.h"
// #include "sim_environment.h"

namespace Aperture {

class Environment;

class Particles : public ParticleBase<single_particle_t>
{
 public:
  typedef ParticleBase<single_particle_t> BaseClass;
  Particles();
  Particles(std::size_t max_num, ParticleType type = ParticleType::electron);
  Particles(const Environment& env, ParticleType type = ParticleType::electron);
  Particles(const Particles& other);
  Particles(Particles&& other);
  virtual ~Particles();

  // void resize(std::size_t max_num);
  // void initialize();
  // void copyFrom(const Particles& other, std::size_t num, std::size_t src_pos = 0, std::size_t dest_pos = 0);
  // void erase(std::size_t pos, std::size_t amount = 1);

  // void copyFrom(const std::vector<single_particle_t>& buffer, std::size_t num, std::size_t src_pos = 0, std::size_t dest_pos = 0);
  // void copyToBuffer(std::vector<single_particle_t>& buffer, std::size_t num, std::size_t src_pos = 0, std::size_t dest_pos = 0);

  using BaseClass::put;
  using BaseClass::append;
  void put(std::size_t pos, Pos_t x, Scalar p, int cell, int flag = 0);
  void append(Pos_t x, Scalar p, int cell, int flag = 0);
  // void put(std::size_t pos, const single_particle_t& part);
  // void swap(Index_t pos, single_particle_t& part);

  // bool is_empty(Index_t pos) const {
  //   return (m_data.cell[pos] == MAX_CELL);
  // }

  // // After rearrange, the index array will all be -1
  // void rearrange(std::vector<Index_t>& index, std::size_t num = 0);
  // // Partition according to a grid configuration, sort the particles
  // // into the bulk part, and those that needs to be communicated out
  // void partition(std::vector<Index_t>& partitions, const Grid& grid);
  // void clear_guard_cells(const Grid& grid);

  // particle_data& data() { return m_data; }
  // const particle_data& data() const { return m_data; }
  ParticleType type() const { return m_type; }
  Scalar charge() const { return m_charge; }
  Scalar mass() const { return m_mass; }
  void set_type(ParticleType type) { m_type = type; }
  void set_charge(Scalar charge) { m_charge = charge; }
  void set_mass(Scalar mass) { m_mass = mass; }
  void track(Index_t pos) { m_data.flag[pos] |= (int)ParticleFlag::tracked; }
  bool check_flag(Index_t pos, ParticleFlag flag) const { return (m_data.flag[pos] & (int)flag) == (int)flag; }
  void set_flag(Index_t pos, ParticleFlag flag) { m_data.flag[pos] |= (int)flag; }

  // The upper 16 bits represent the rank the particle is born
  // int tag_rank(Index_t idx) { return (m_data.tag_id[idx] >> 16); }
  // The lower 16 bits represent the id of the particle tracked
  // int tag_id(Index_t idx) { return (m_data.tag_id[idx] & ((1 << 16) - 1)); }

 private:

  // char* m_data_ptr;
  // particle_data m_data;
  ParticleType m_type;
  Scalar m_charge = 1.0;
  Scalar m_mass = 1.0;

  // std::vector<Index_t> m_index;
}; // ----- end of class Particles : public ParticleBase -----

}


#endif   // _PARTICLES_H_
