#ifndef _PARTICLES_1D_H_
#define _PARTICLES_1D_H_

#include "cuda/data/particle_base_dev.h"
#include "utils/util_functions.h"
#include <cstdlib>
#include <string>
#include <vector>
// #include "core/grid.h"
#include "core/constant_defs.h"

namespace Aperture {

struct SimParams;

class Particles_1D : public particle_base_dev<single_particle1d_t> {
 public:
  typedef particle_base_dev<single_particle1d_t> BaseClass;
  typedef particle1d_data DataClass;
  Particles_1D();
  Particles_1D(std::size_t max_num);
  Particles_1D(const SimParams& env);
  Particles_1D(const Particles_1D& other);
  Particles_1D(Particles_1D&& other);
  virtual ~Particles_1D();

  using BaseClass::append;
  using BaseClass::put;
  // void put(std::size_t pos, Pos_t x1, Scalar p1, int cell,
  //          ParticleType type, Scalar weight = 1.0, uint32_t flag = 0);
  // void append(Pos_t x1, Scalar p1, int cell, ParticleType type,
  //             Scalar weight = 1.0, uint32_t flag = 0);

 private:
  std::vector<Index_t> m_partition;
};  // ----- end of class Particles_1D : public
    // particle_base<single_particle1d_t> -----

}  // namespace Aperture

#endif  // _PARTICLES_1D_H_
