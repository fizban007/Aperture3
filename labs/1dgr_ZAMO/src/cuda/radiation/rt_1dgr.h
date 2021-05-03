#ifndef _RT_1DGR_H_
#define _RT_1DGR_H_

#include "core/typedefs.h"
#include "core/array.h"
#include "core/fields.h"
#include "cuda/radiation/rt_ic.h"

namespace Aperture {

class sim_environment;
struct sim_data;

class RadiationTransfer1DGR {
 public:
  RadiationTransfer1DGR(sim_environment& env);
  virtual ~RadiationTransfer1DGR();

  void emit_photons(sim_data& data, Scalar dt);
  void produce_pairs(sim_data& data, Scalar dt);

 private:
  sim_environment& m_env;
  void* d_rand_states;
  int m_threadsPerBlock, m_blocksPerGrid;
  array<int> m_numPerBlock;
  array<int> m_cumNumPerBlock;
  array<int> m_posInBlock;
  inverse_compton m_ic;
};  // ----- end of class RadiationTransfer1DGR -----

}  // namespace Aperture

#endif  // _RT_1DGR_H_
