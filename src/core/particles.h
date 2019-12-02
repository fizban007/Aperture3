#ifndef _PARTICLES_H_
#define _PARTICLES_H_

#include "particle_base.h"
#include "utils/util_functions.h"

namespace Aperture {

class particles_t : public particle_base<single_particle_t> {
 public:
  typedef particle_base<single_particle_t> base_class;
  typedef particle_data data_class;

  particles_t();
  particles_t(size_t max_num, bool managed = false);
  particles_t(const particles_t& other);
  particles_t(particles_t&& other);
  virtual ~particles_t();

  using base_class::append;
  using base_class::put;
  void put(std::size_t pos, const Vec3<Pos_t>& x, const Vec3<Scalar>& p,
           int cell, ParticleType type, Scalar weight = 1.0,
           uint32_t flag = 0);
  void append(const Vec3<Pos_t>& x, const Vec3<Scalar>& p, int cell,
              ParticleType type, Scalar weight = 1.0,
              uint32_t flag = 0);
  void compute_energies();

  // using base_class::compute_spectrum;
  void compute_spectrum(int num_bins, std::vector<Scalar>& energies,
                        std::vector<uint32_t>& nums, ParticleFlag flag);

};  // ----- end of class particles_t : public

}  // namespace Aperture

#endif  // _PARTICLES_H_
