#ifndef _RT_MAGNETAR_H_
#define _RT_MAGNETAR_H_

#include "cuda/data/array.h"
#include "cuda/data/fields_dev.h"
#include "core/typedefs.h"

namespace Aperture {

class cu_sim_environment;
class Particles;
class Photons;
struct cu_sim_data;

class RadiationTransferMagnetar {
 public:
  RadiationTransferMagnetar(const cu_sim_environment& env);
  virtual ~RadiationTransferMagnetar();

  void emit_photons(cu_sim_data& data);
  void produce_pairs(cu_sim_data& data);

  cu_scalar_field<Scalar>& get_pair_events() { return m_pair_events; }
  cu_scalar_field<Scalar>& get_ph_events() { return m_ph_events; }

 private:
  const cu_sim_environment& m_env;
  void* d_rand_states;
  int m_threadsPerBlock, m_blocksPerGrid;
  cu_array<int> m_numPerBlock;
  cu_array<int> m_cumNumPerBlock;
  cu_array<int> m_posInBlock;

  cu_scalar_field<Scalar> m_pair_events;
  cu_scalar_field<Scalar> m_ph_events;
};  // ----- end of class RadiationTransferPulsar -----


}

#endif  // _RT_MAGNETAR_H_
