#ifndef _RADIATION_TRANSFER_H_
#define _RADIATION_TRANSFER_H_

#include "data/typedefs.h"

namespace Aperture {

class Environment;

template <typename PtcClass, typename PhotonClass>
class RadiationTransfer
{
 public:
  RadiationTransfer(const Environment& env);
  virtual ~RadiationTransfer();

  void emit_photons(PhotonClass& photons, PtcClass& ptc);
  void produce_pairs(PtcClass& ptc, PhotonClass& photons);

 private:
  const Environment& m_env;
  void* d_rand_states;
  int m_threadsPerBlock, m_blocksPerGrid;
}; // ----- end of class RadiationTransfer -----



}

#endif  // _RADIATION_TRANSFER_H_
