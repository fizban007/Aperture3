#ifndef _PARTICLES_1D_H_
#define _PARTICLES_1D_H_

#include <cstdlib>
#include <vector>
#include <string>
#include "utils/util_functions.h"
#include "data/particle_base.h"
// #include "data/grid.h"
#include "constant_defs.h"

namespace Aperture {

class SimParams;

class Particles_1D : public ParticleBase<single_particle1d_t>
{
 public:
  typedef ParticleBase<single_particle1d_t> BaseClass;
  typedef particle1d_data DataClass;
  Particles_1D();
  Particles_1D(std::size_t max_num);
  Particles_1D(const SimParams& env);
  Particles_1D(const Particles_1D& other);
  Particles_1D(Particles_1D&& other);
  virtual ~Particles_1D();

  using BaseClass::put;
  using BaseClass::append;
  void put(std::size_t pos, Pos_t x1, Scalar p1, int cell, ParticleType type,
           Scalar weight = 1.0, uint32_t flag = 0);
  void append(Pos_t x1, Scalar p1, int cell, ParticleType type,
              Scalar weight = 1.0, uint32_t flag = 0);
  void track(Index_t pos) { m_data.flag[pos] |= (int)ParticleFlag::tracked; }
  bool check_flag(Index_t pos, ParticleFlag flag) const { return (m_data.flag[pos] & (unsigned int)flag) == (unsigned int)flag; }
  void set_flag(Index_t pos, ParticleFlag flag) { m_data.flag[pos] |= (unsigned int)flag; }
  // Use the highest 3 bits to represent particle type
  ParticleType check_type(Index_t pos) const { return (ParticleType)get_ptc_type(m_data.flag[pos]); }
  void set_type(Index_t pos, ParticleType type) {
    m_data.flag[pos] = set_ptc_type_flag(m_data.flag[pos], type);
  }

 private:
  std::vector<Index_t> m_partition;
}; // ----- end of class Particles_1D : public ParticleBase<single_particle1d_t> -----


}

#endif  // _PARTICLES_1D_H_
