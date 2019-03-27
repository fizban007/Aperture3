#ifndef _PARTICLES_DEV_H_
#define _PARTICLES_DEV_H_

#include "core/constant_defs.h"
#include "core/grid.h"
#include "cuda/data/cu_multi_array.h"
#include "cuda/data/particle_base_dev.h"
#include "utils/util_functions.h"
#include <cstdlib>
#include <string>
#include <vector>
// #include "sim_environment.h"

namespace Aperture {

struct SimParams;
struct cu_sim_data;

class Particles : public particle_base_dev<single_particle_t> {
 public:
  typedef particle_base_dev<single_particle_t> BaseClass;
  typedef particle_data DataClass;
  Particles();
  Particles(std::size_t max_num);
  Particles(const SimParams& params);
  Particles(const Particles& other) = delete;
  Particles(Particles&& other);
  virtual ~Particles();

  // void resize(std::size_t max_num);
  // void initialize();
  // void copy_from(const Particles& other, std::size_t num, std::size_t
  // src_pos = 0, std::size_t dest_pos = 0); void erase(std::size_t pos,
  // std::size_t amount = 1);

  // void copy_from(const std::vector<single_particle_t>& buffer,
  // std::size_t num, std::size_t src_pos = 0, std::size_t dest_pos =
  // 0); void copyToBuffer(std::vector<single_particle_t>& buffer,
  // std::size_t num, std::size_t src_pos = 0, std::size_t dest_pos =
  // 0);

  using BaseClass::append;
  using BaseClass::put;
  void put(std::size_t pos, const Vec3<Pos_t>& x, const Vec3<Scalar>& p,
           int cell, ParticleType type, Scalar weight = 1.0,
           uint32_t flag = 0);
  void append(const Vec3<Pos_t>& x, const Vec3<Scalar>& p, int cell,
              ParticleType type, Scalar weight = 1.0,
              uint32_t flag = 0);
  void compute_energies();

  using BaseClass::compute_spectrum;
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

  // The upper 16 bits represent the rank the particle is born
  // int tag_rank(Index_t idx) { return (m_data.tag_id[idx] >> 16); }
  // The lower 16 bits represent the id of the particle tracked
  // int tag_id(Index_t idx) { return (m_data.tag_id[idx] & ((1 << 16) -
  // 1)); }

 private:
  // char* m_data_ptr;
  // particle_data m_data;
  // ParticleType m_type;
  // Scalar m_charge = 1.0;
  // Scalar m_mass = 1.0;
  // std::vector<Index_t> m_partition;
  // cu_multi_array<Scalar> m_dens;

  // std::vector<Index_t> m_index;
};  // ----- end of class Particles : public particle_base_dev -----

}  // namespace Aperture

#endif  // _PARTICLES_DEV_H_
