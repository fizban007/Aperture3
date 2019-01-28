#ifndef _RADIATION_TRANSFER_H_
#define _RADIATION_TRANSFER_H_

#include "data/array.h"
#include "core/typedefs.h"
#include "data/fields_dev.h"

namespace Aperture {

class cu_sim_environment;

template <typename PtcClass, typename PhotonClass, typename RadModel>
class RadiationTransfer {
 public:
  RadiationTransfer(const cu_sim_environment& env);
  virtual ~RadiationTransfer();

  void emit_photons(PhotonClass& photons, PtcClass& ptc);
  void produce_pairs(PtcClass& ptc, PhotonClass& photons);

  cu_scalar_field<Scalar>& get_pair_events() { return m_pair_events; }
  cu_scalar_field<Scalar>& get_ph_events() { return m_ph_events; }

 private:
  const cu_sim_environment& m_env;
  void* d_rand_states;
  int m_threadsPerBlock, m_blocksPerGrid;
  Array<int> m_numPerBlock;
  Array<int> m_cumNumPerBlock;
  Array<int> m_posInBlock;

  cu_scalar_field<Scalar> m_pair_events;
  cu_scalar_field<Scalar> m_ph_events;
};  // ----- end of class RadiationTransfer -----

}  // namespace Aperture

#endif  // _RADIATION_TRANSFER_H_
