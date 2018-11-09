#ifndef _RADIATION_TRANSFER_PULSAR_H_
#define _RADIATION_TRANSFER_PULSAR_H_

#include "data/array.h"
#include "data/fields.h"
#include "data/typedefs.h"

namespace Aperture {

class Environment;
class Particles;
class Photons;

class RadiationTransferPulsar {
 public:
  RadiationTransferPulsar(const Environment& env);
  virtual ~RadiationTransferPulsar();

  void emit_photons(Photons& photons, Particles& ptc);
  void produce_pairs(Particles& ptc, Photons& photons);

  ScalarField<Scalar>& get_pair_events() { return m_pair_events; }
  ScalarField<Scalar>& get_ph_events() { return m_ph_events; }

 private:
  const Environment& m_env;
  void* d_rand_states;
  int m_threadsPerBlock, m_blocksPerGrid;
  Array<int> m_numPerBlock;
  Array<int> m_cumNumPerBlock;
  Array<int> m_posInBlock;

  ScalarField<Scalar> m_pair_events;
  ScalarField<Scalar> m_ph_events;
};  // ----- end of class RadiationTransferPulsar -----

}  // namespace Aperture

#endif  // _RADIATION_TRANSFER_H_
