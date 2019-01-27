#ifndef _RT_1D_H_
#define _RT_1D_H_

// #include "data/array.h"
#include <random>

namespace Aperture {

class sim_environment;
struct sim_data;

class rad_transfer_1d {
 public:
  rad_transfer_1d(const sim_environment& env);
  virtual ~rad_transfer_1d();

  void emit_photons(sim_data& data);
  void produce_pairs(sim_data& data);

 private:
  const sim_environment& m_env;
  std::default_random_engine m_gen;
  std::uniform_real_distribution<double> m_dist;
  std::normal_distribution<double> m_normal;
};  // ----- end of class rad_transfer_1d -----

}  // namespace Aperture

#endif  // _RT_1D_H_
