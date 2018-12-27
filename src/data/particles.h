#ifndef _PARTICLES_H_
#define _PARTICLES_H_

#include "data/enum_types.h"
#include "data/particle_base.h"
#include "utils/util_functions.h"

namespace Aperture {

struct SimParams;

class particles_t : public ParticleBase<single_particle_t>
{
 public:
  typedef ParticleBase<single_particle_t> base_class;
  typedef particle_data data_class;
  
  particles_t();
  particles_t(size_t max_num);
  particles_t(const SimParams& params);
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

  using base_class::compute_spectrum;
  void compute_spectrum(int num_bins, std::vector<Scalar>& energies,
                        std::vector<uint32_t>& nums, ParticleFlag flag);

  void track(Index_t pos) {
    m_data.flag[pos] |= (int)ParticleFlag::tracked;
  }
  bool check_flag(Index_t pos, ParticleFlag flag) const {
    return (m_data.flag[pos] & (unsigned int)flag) ==
           (unsigned int)flag;
  }
  void set_flag(Index_t pos, ParticleFlag flag) {
    m_data.flag[pos] |= (unsigned int)flag;
  }
  // Use the highest 3 bits to represent particle type
  ParticleType check_type(Index_t pos) const {
    return (ParticleType)get_ptc_type(m_data.flag[pos]);
  }
  void set_type(Index_t pos, ParticleType type) {
    m_data.flag[pos] = set_ptc_type_flag(m_data.flag[pos], type);
  }

 private:
  
}; // ----- end of class particles_t : public ParticleBase<single_particle_t> -----


}

#endif  // _PARTICLES_H_
