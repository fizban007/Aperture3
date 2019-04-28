#ifndef _RADIATION_TRANSFER_PULSAR_H_
#define _RADIATION_TRANSFER_PULSAR_H_

#include "cuda/data/array.h"
#include "cuda/data/fields_dev.h"
#include "core/typedefs.h"

namespace Aperture {

class cu_sim_environment;
struct cu_sim_data;

class RadiationTransferPulsar {
 public:
  RadiationTransferPulsar(const cu_sim_environment& env);
  virtual ~RadiationTransferPulsar();

  void emit_photons(cu_sim_data& data);
  void produce_pairs(cu_sim_data& data);

 private:
  const cu_sim_environment& m_env;
  std::vector<void*> d_rand_states;
  int m_threadsPerBlock, m_blocksPerGrid;
  std::vector<cu_array<int>> m_numPerBlock;
  std::vector<cu_array<int>> m_cumNumPerBlock;
  std::vector<cu_array<int>> m_posInBlock;
};  // ----- end of class RadiationTransferPulsar -----

}  // namespace Aperture

#endif  // _RADIATION_TRANSFER_H_
