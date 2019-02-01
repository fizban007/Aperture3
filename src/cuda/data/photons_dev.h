#ifndef _PHOTONS_DEV_H_
#define _PHOTONS_DEV_H_

#include <cstdlib>
#include <random>
#include <string>
#include <vector>
// #include "data/particles_dev.h"
#include "cuda/data/particle_base_dev.h"
#include "core/quadmesh.h"

namespace Aperture {

class cu_sim_environment;
struct SimParams;

class Photons : public particle_base_dev<single_photon_t> {
 public:
  typedef particle_base_dev<single_photon_t> BaseClass;
  typedef photon_data DataClass;
  Photons();
  Photons(std::size_t max_num);
  Photons(const cu_sim_environment& env);
  Photons(const SimParams& params);
  Photons(const Photons& other);
  Photons(Photons&& other);
  virtual ~Photons();

  using BaseClass::append;
  using BaseClass::put;
  void put(std::size_t pos, const Vec3<Pos_t>& x, const Vec3<Scalar>& p,
           Scalar path_left, int cell, Scalar weight = 1.0,
           uint32_t flag = 0);
  void append(const Vec3<Pos_t>& x, const Vec3<Scalar>& p,
              Scalar path_left, int cell, Scalar weight = 1.0,
              uint32_t flag = 0);

  // void convert_pairs(Particles& electrons, Particles& positrons);
  // void emit_photons(Particles& electrons, Particles& positrons, const
  // Quadmesh& mesh); void move(const Grid& grid, double dt);
  void sort(const Grid& grid);
  void compute_energies();

  void track(Index_t pos) {
    m_data.flag[pos] |= (int)ParticleFlag::tracked;
  }
  bool check_flag(Index_t pos, PhotonFlag flag) const {
    return (m_data.flag[pos] & (unsigned int)flag) ==
           (unsigned int)flag;
  }
  void set_flag(Index_t pos, PhotonFlag flag) {
    m_data.flag[pos] |= (unsigned int)flag;
  }

 private:
  std::vector<Index_t> m_partition;
}; // class Photons

}  // namespace Aperture

// #include "types/detail/photons_impl.hpp"

#endif  // _PHOTONS_DEV_H_