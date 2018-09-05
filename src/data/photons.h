#ifndef _PHOTONS_H_
#define _PHOTONS_H_

#include <cstdlib>
#include <vector>
#include <string>
#include <random>
// #include "data/particles.h"
#include "data/particle_base.h"
#include "data/quadmesh.h"

namespace Aperture {

class Environment;

class Photons : public ParticleBase<single_photon_t>
{
 public:
  typedef ParticleBase<single_photon_t> BaseClass;
  typedef photon_data DataClass;
  Photons();
  Photons(std::size_t max_num);
  Photons(const Environment& env);
  Photons(const Photons& other);
  Photons(Photons&& other);
  virtual ~Photons();

  using BaseClass::put;
  using BaseClass::append;
  void put(std::size_t pos, const Vec3<Pos_t>& x, const Vec3<Scalar>& p,
           Scalar path_left, int cell, Scalar weight = 1.0, uint32_t flag = 0);
  void append(const Vec3<Pos_t>& x, const Vec3<Scalar>& p,
              Scalar path_left, int cell, Scalar weight = 1.0, uint32_t flag = 0);

  // void convert_pairs(Particles& electrons, Particles& positrons);
  // void emit_photons(Particles& electrons, Particles& positrons, const Quadmesh& mesh);
  // void move(const Grid& grid, double dt);
  void sort(const Grid& grid);

  void track(Index_t pos) { m_data.flag[pos] |= (int)ParticleFlag::tracked; }
  bool check_flag(Index_t pos, PhotonFlag flag) const { return (m_data.flag[pos] & (unsigned int)flag) == (unsigned int)flag; }
  void set_flag(Index_t pos, PhotonFlag flag) { m_data.flag[pos] |= (unsigned int)flag; }

 private:
  std::vector<Index_t> m_partition;

};

}

// #include "types/detail/photons_impl.hpp"

#endif  // _PHOTONS_H_
