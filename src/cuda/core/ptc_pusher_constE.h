#ifndef _PTC_PUSHER_CONSTE_H_
#define _PTC_PUSHER_CONSTE_H_

#include "particle_pusher.h"

namespace Aperture {

class cu_sim_environment;

class ParticlePusher_ConstE : public ParticlePusher {
 public:
  typedef ParticlePusher_ConstE self_type;

  ParticlePusher_ConstE(const cu_sim_environment& env);
  virtual ~ParticlePusher_ConstE();

  virtual void push(cu_sim_data& data, double dt);
  virtual void handle_boundary(cu_sim_data& data);

 private:
  Scalar m_E;
}; // ----- end of class ParticlePusher_ConstE : public ParticlePusher -----


}

#endif  // _PTC_PUSHER_CONSTE_H_
