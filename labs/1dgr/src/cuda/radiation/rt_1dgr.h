#ifndef _RT_1DGR_H_
#define _RT_1DGR_H_

#include "core/typedefs.h"
#include "cuda/data/array.h"
#include "cuda/data/fields_dev.h"
#include "cuda/radiation/rt_ic.h"

namespace Aperture {

class cu_sim_environment;
struct cu_sim_data;

class RadiationTransfer1DGR {
 public:
  RadiationTransfer1DGR(const cu_sim_environment& env);
  virtual ~RadiationTransfer1DGR();

  void emit_photons(cu_sim_data& data, Scalar dt);
  void produce_pairs(cu_sim_data& data, Scalar dt);

 private:
  const cu_sim_environment& m_env;
  void* d_rand_states;
  int m_threadsPerBlock, m_blocksPerGrid;
  cu_array<int> m_numPerBlock;
  cu_array<int> m_cumNumPerBlock;
  cu_array<int> m_posInBlock;
  inverse_compton m_ic;
};  // ----- end of class RadiationTransfer1DGR -----

}  // namespace Aperture

#endif  // _RT_1DGR_H_
