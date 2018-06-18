#ifndef _INVERSE_COMPTON_H_
#define _INVERSE_COMPTON_H_

#include "data/photons.h"
#include "data/particles.h"

// class curandState;

namespace Aperture {

class Environment;

class InverseCompton
{
 public:
  InverseCompton(const Environment& env);
  virtual ~InverseCompton();

  void convert_pairs(Particles& particles, Photons& photons);
  void emit_photons(Photons& photons, Particles& particles);

 private:
  const Environment& m_env;
  void* d_rand_states;
  int m_threadsPerBlock, m_blocksPerGrid;
}; // ----- end of class Inverse -----


}

#endif  // _INVERSE_COMPTON_H_
