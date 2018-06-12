#ifndef _PHOTONS_H_
#define _PHOTONS_H_

#include <cstdlib>
#include <vector>
#include <string>
#include <random>
#include "data/particles.h"
#include "data/quadmesh.h"

namespace Aperture {

class Environment;

class Photons : public ParticleBase<single_photon_t>
{
 public:
  Photons();
  Photons(std::size_t max_num);
  Photons(const Environment& env);
  Photons(const Photons& other);
  Photons(Photons&& other);
  virtual ~Photons();

  void put(std::size_t pos, Pos_t x, Scalar p, Scalar path_left, int cell, int flag = 0);
  void append(Pos_t x, Scalar p, Scalar path_left, int cell, int flag = 0);

  void convert_pairs(Particles& electrons, Particles& positrons);
  void emit_photons(Particles& electrons, Particles& positrons, const Quadmesh& mesh);
  void move(const Grid& grid, double dt);
  void sort(const Grid& grid);

  bool check_flag(Index_t pos, PhotonFlag flag) const { return (m_data.flag[pos] & (unsigned int)flag) == (unsigned int)flag; }
  void set_flag(Index_t pos, PhotonFlag flag) { m_data.flag[pos] |= (unsigned int)flag; }

  void compute_A1(double er);
  void compute_A2(double er, double et);
  double f_inv1(double u, double gamma);
  double f_inv2(double u, double gamma);
  double draw_photon_e1p(double gamma);
  double draw_photon_ep(double e1p, double gamma);
  double draw_photon_u1p(double e1p, double gamma);
  double draw_photon_energy(double gamma, double p, double x);
  double draw_photon_freepath(double Eph);

 private:
  bool create_pairs = false;
  bool trace_photons = false;
  float gamma_thr = 10.0;
  float l_ph = 1.0;
  float p_ph = 1.0;
  float p_ic = 1.0;
  float track_pct = 0.2;

  float alpha = 2.0;
  float e_s = 0.2;
  float e_min = 1.0e-3;
  double A1;
  double A2;
  std::vector<Index_t> m_partition;

  std::default_random_engine m_generator;
  std::uniform_real_distribution<float> m_dist;
  std::normal_distribution<float> m_normal;


};

}

// #include "types/detail/photons_impl.hpp"

#endif  // _PHOTONS_H_
