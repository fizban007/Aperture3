#ifndef _PHOTONS_1D_H_
#define _PHOTONS_1D_H_

#include <cstdlib>
#include <random>
#include <string>
#include <vector>
// #include "data/particles_dev.h"
#include "cuda/data/particle_base_dev.h"
#include "core/quadmesh.h"

namespace Aperture {

struct SimParams;

class Photons_1D : public particle_base_dev<single_photon1d_t> {
 public:
  typedef particle_base_dev<single_photon1d_t> BaseClass;
  typedef photon1d_data DataClass;
  Photons_1D();
  Photons_1D(std::size_t max_num);
  Photons_1D(const SimParams& params);
  Photons_1D(const Photons_1D& other);
  Photons_1D(Photons_1D&& other);
  virtual ~Photons_1D();

  using BaseClass::append;
  using BaseClass::put;
  // void put(std::size_t pos, Pos_t x1, Scalar p1, Scalar path_left,
  //          int cell, Scalar weight = 1.0, uint32_t flag = 0);
  // void append(Pos_t x1, Scalar p1, Scalar path_left, int cell,
  //             Scalar weight = 1.0, uint32_t flag = 0);

  // void convert_pairs(Particles& electrons, Particles& positrons);
  // void emit_photons(Particles& electrons, Particles& positrons, const
  // Quadmesh& mesh); void move(const Grid& grid, double dt);
  // void sort(const Grid& grid);

  // void track(Index_t pos) {
  //   m_data.flag[pos] |= (int)ParticleFlag::tracked;
  // }
  // bool check_flag(Index_t pos, PhotonFlag flag) const {
  //   return (m_data.flag[pos] & (unsigned int)flag) ==
  //          (unsigned int)flag;
  // }
  // void set_flag(Index_t pos, PhotonFlag flag) {
  //   m_data.flag[pos] |= (unsigned int)flag;
  // }

 private:
  std::vector<Index_t> m_partition;
};

}  // namespace Aperture

#endif  // _PHOTONS_1D_H_
