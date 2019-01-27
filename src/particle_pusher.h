#ifndef _PARTICLE_PUSHER_H_
#define _PARTICLE_PUSHER_H_

// #include "core/callbacks.h"
#include "cu_sim_data.h"

namespace Aperture {

class ParticlePusher {
 public:
  typedef cu_vector_field<Scalar> vfield;
  typedef cu_scalar_field<Scalar> sfield;
  typedef ParticlePusher self_type;

  ParticlePusher() {}
  virtual ~ParticlePusher() {}

  virtual void push(cu_sim_data& data, double dt) = 0;
  virtual void handle_boundary(cu_sim_data& data) = 0;
  // virtual void push(Particles& particles, const vfield_t& E, const
  // vfield_t& B, double dt) = 0;

  // void register_ptc_comm_callback(const ptc_comm_callback& callback)
  // {
  //   m_comm = callback;
  // }

  // virtual void print() = 0;

 protected:
  // ptc_comm_callback m_comm;
};  // ----- end of class particle_pusher -----
}  // namespace Aperture

#endif  // _PARTICLE_PUSHER_H_
