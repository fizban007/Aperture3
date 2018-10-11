#ifndef _PTC_PUSHER_CONSTE_H_
#define _PTC_PUSHER_CONSTE_H_

#include "particle_pusher.h"

namespace Aperture {

class Environment;

class ParticlePusher_ConstE : public ParticlePusher {
 public:
  typedef ParticlePusher_ConstE self_type;

  ParticlePusher_ConstE(const Environment& env);
  virtual ~ParticlePusher_ConstE();

  virtual void push(SimData& data, double dt);
  virtual void handle_boundary(SimData& data);

 private:
  Scalar m_E;
}; // ----- end of class ParticlePusher_ConstE : public ParticlePusher -----


}

#endif  // _PTC_PUSHER_CONSTE_H_
