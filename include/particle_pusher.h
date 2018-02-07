#ifndef _PARTICLE_PUSHER_H_
#define _PARTICLE_PUSHER_H_

// #include "data/callbacks.h"
#include "sim_data.h"

namespace Aperture {

class ParticlePusher {
 public:
  typedef VectorField<Scalar> vfield;
  typedef ScalarField<Scalar> sfield;
  typedef ParticlePusher self_type;

  ParticlePusher() {}
  virtual ~ParticlePusher() {}

  virtual void push(SimData& data, double dt) = 0;
  virtual void handle_boundary(SimData& data) = 0;
  // virtual void push(Particles& particles, const vfield_t& E, const vfield_t& B, double dt) = 0;

  // void register_ptc_comm_callback(const ptc_comm_callback& callback) {
  //   m_comm = callback;
  // }

  self_type& set_algorithm(ForceAlgorithm algorithm) {
    m_algorithm = algorithm;
    return *this;
  }

  self_type& set_gravity(double g) {
    m_g = g;
    return *this;
  }

  self_type& set_radiation(bool radiation) {
    m_radiation = radiation;
    return *this;
  }

  self_type& set_compute_curvature(bool c) {
    m_compute_curvature = c;
    return *this;
  }

  self_type& set_periodic(bool p) {
    m_periodic = p;
    return *this;
  }

  self_type& set_interp_order(int n) {
    m_interp = n;
    return *this;
  }

  // virtual void print() = 0;

 protected:
  // ptc_comm_callback m_comm;
  bool m_gravity, m_radiation, m_compute_curvature, m_periodic;
  ForceAlgorithm m_algorithm;
  int m_interp = 1;

  // Lorentz_force_Boris m_boris;
  // Lorentz_force_Vay m_vay;
  double m_g = 0;
};  // ----- end of class particle_pusher -----
}

#endif  // _PARTICLE_PUSHER_H_
