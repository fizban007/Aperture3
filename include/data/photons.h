#ifndef _PHOTONS_H_
#define _PHOTONS_H_

#include <cstdlib>
#include <vector>
#include <string>
#include "data/particles.h"

namespace Aperture {

class Photons : public ParticleBase<single_photon_t>
{
 public:
  Photons();
  Photons(std::size_t max_num);
  Photons(const Environment& env);
  Photons(const Photons& other);
  Photons(Photons&& other);
  virtual ~Photons();

  void put(std::size_t pos, Pos_t x, Scalar p, int cell, int flag = 0);
  void append(Pos_t x, Scalar p, int cell, int flag = 0);

  void convert_pairs(Particles& electrons, Particles& positrons);

  // bool check_flag(Index_t pos, PhotonFlag flag) { return (m_data.flag[pos] & (int)flag) == (int)flag; }
  // void set_flag(Index_t pos, PhotonFlag flag) { m_data.flag[pos] |= (int)flag; }

 private:
  void make_pair(Index_t pos, Particles& electrons, Particles& positrons);
};

}

// #include "types/detail/photons_impl.hpp"

#endif  // _PHOTONS_H_
