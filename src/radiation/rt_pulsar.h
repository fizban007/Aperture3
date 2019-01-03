#ifndef _RADIATION_TRANSFER_PULSAR_H_
#define _RADIATION_TRANSFER_PULSAR_H_

#include "data/array.h"
#include "data/fields_dev.h"
#include "data/typedefs.h"

namespace Aperture {

class Environment;
class Particles;
class Photons;
struct SimData;

class RadiationTransferPulsar {
 public:
  RadiationTransferPulsar(const Environment& env);
  virtual ~RadiationTransferPulsar();

  void emit_photons(SimData& data);
  void produce_pairs(SimData& data);

  cu_scalar_field<Scalar>& get_pair_events() { return m_pair_events; }
  cu_scalar_field<Scalar>& get_ph_events() { return m_ph_events; }

 private:
  const Environment& m_env;
  void* d_rand_states;
  int m_threadsPerBlock, m_blocksPerGrid;
  Array<int> m_numPerBlock;
  Array<int> m_cumNumPerBlock;
  Array<int> m_posInBlock;

  cu_scalar_field<Scalar> m_pair_events;
  cu_scalar_field<Scalar> m_ph_events;
};  // ----- end of class RadiationTransferPulsar -----

}  // namespace Aperture

#endif  // _RADIATION_TRANSFER_H_
