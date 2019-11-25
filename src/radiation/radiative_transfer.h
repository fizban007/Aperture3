#ifndef _RADIATIVE_TRANSFER_H_
#define _RADIATIVE_TRANSFER_H_

#include "core/array.h"

namespace Aperture {

class sim_environment;
struct sim_data;

class radiative_transfer {
 public:
  radiative_transfer(sim_environment& env);
  ~radiative_transfer();

  void initialize();
  void emit_photons(sim_data& data);
  void produce_pairs(sim_data& data);

 protected:
  sim_environment& m_env;

  // These are helper data for the cuda implementation
  int m_threadsPerBlock, m_blocksPerGrid;
  array<int> m_numPerBlock;
  array<int> m_cumNumPerBlock;
  array<int> m_posInBlock;
};

}

#endif  // _RADIATIVE_TRANSFER_H_
